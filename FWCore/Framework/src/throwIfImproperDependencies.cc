// -*- ++ -*-
//
// Package:     FWCore/Framework
// Function:    throwIfImproperDependencies
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  root
//         Created:  Tue, 06 Sep 2016 16:04:28 GMT
//

// system include files

// user include files
#include "FWCore/Framework/src/throwIfImproperDependencies.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "boost/graph/graph_traits.hpp"
#include "boost/graph/adjacency_list.hpp"
#include "boost/graph/depth_first_search.hpp"
#include "boost/graph/visitors.hpp"

namespace {
  //====================================
  // checkForCorrectness algorithm
  //
  // The code creates a 'dependency' graph between all
  // modules. A module depends on another module if
  // 1) it 'consumes' data produced by that module
  // 2) it appears directly after the module within a Path
  //
  // If there is a cycle in the 'dependency' graph then
  // the schedule may be unrunnable. The schedule is still
  // runnable if all cycles have at least two edges which
  // connect modules only by Path dependencies (i.e. not
  // linked by a data dependency).
  //
  //  Example 1:
  //  C consumes data from B
  //  Path 1: A + B + C
  //  Path 2: B + C + A
  //
  //  Cycle: A after C [p2], C consumes B, B after A [p1]
  //  Since this cycle has 2 path only edges it is OK since
  //  A and (B+C) are independent so their run order doesn't matter
  //
  //  Example 2:
  //  B consumes A
  //  C consumes B
  //  Path: C + A
  //
  //  Cycle: A after C [p], C consumes B, B consumes A
  //  Since this cycle has 1 path only edge it is unrunnable.
  //
  //  Example 3:
  //  A consumes B
  //  B consumes C
  //  C consumes A
  //  (no Path since unscheduled execution)
  //
  //  Cycle: A consumes B, B consumes C, C consumes A
  //  Since this cycle has 0 path only edges it is unrunnable.
  //====================================

  typedef std::pair<unsigned int, unsigned int> SimpleEdge;
  typedef std::map<SimpleEdge, std::vector<unsigned int>> EdgeToPathMap;

  typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS> Graph;

  typedef boost::graph_traits<Graph>::edge_descriptor Edge;

  template <typename T>
  std::unordered_set<T> intersect(std::unordered_set<T> const& iLHS, std::unordered_set<T> const& iRHS) {
    std::unordered_set<T> result;
    if (iLHS.size() < iRHS.size()) {
      result.reserve(iLHS.size());
      for (auto const& l : iLHS) {
        if (iRHS.find(l) != iRHS.end()) {
          result.insert(l);
        }
      }
      return result;
    }
    result.reserve(iRHS.size());
    for (auto const& r : iRHS) {
      if (iLHS.find(r) != iLHS.end()) {
        result.insert(r);
      }
    }
    return result;
  }

  struct cycle_detector : public boost::dfs_visitor<> {
    static const unsigned int kRootVertexIndex = 0;

    cycle_detector(EdgeToPathMap const& iEdgeToPathMap,
                   std::vector<std::vector<unsigned int>> const& iPathIndexToModuleIndexOrder,
                   std::vector<std::string> const& iPathNames,
                   std::unordered_map<unsigned int, std::string> const& iModuleIndexToNames)
        : m_edgeToPathMap(iEdgeToPathMap),
          m_pathIndexToModuleIndexOrder(iPathIndexToModuleIndexOrder),
          m_pathNames(iPathNames),
          m_indexToNames(iModuleIndexToNames) {}

    bool compare(Edge const& iLHS, Edge const& iRHS) const;

    void tree_edge(Edge const& iEdge, Graph const& iGraph) {
      auto const& index = get(boost::vertex_index, iGraph);

      auto in = index[source(iEdge, iGraph)];
      for (auto it = m_stack.begin(); it != m_stack.end(); ++it) {
        if (in == index[source(*it, iGraph)]) {
          //this vertex is now being used to probe a new edge
          // so we should drop the rest of the tree
          m_stack.erase(it, m_stack.end());
          break;
        }
      }

      m_stack.push_back(iEdge);
    }

    void finish_vertex(unsigned int iVertex, Graph const& iGraph) {
      if (not m_stack.empty()) {
        auto const& index = get(boost::vertex_index, iGraph);

        if (iVertex == index[source(m_stack.back(), iGraph)]) {
          m_stack.pop_back();
        }
      }
    }

    //Called if a cycle happens
    void back_edge(Edge const& iEdge, Graph const& iGraph) {
      auto const& index = get(boost::vertex_index, iGraph);

      if (kRootVertexIndex != index[source(m_stack.front(), iGraph)]) {
        //this part of the graph is not connected to data processing
        return;
      }

      m_stack.push_back(iEdge);

      auto pop_stack = [](std::vector<Edge>* stack) { stack->pop_back(); };
      std::unique_ptr<std::vector<Edge>, decltype(pop_stack)> guard(&m_stack, pop_stack);

      //This edge has not been added to the stack yet
      // making a copy allows us to add it in but not worry
      // about removing it at the end of the routine
      std::vector<Edge> tempStack;

      tempStack = findMinimumCycle(m_stack, iGraph);
      checkCycleForProblem(tempStack, iGraph);
      for (auto const& edge : tempStack) {
        unsigned int in = index[source(edge, iGraph)];
        unsigned int out = index[target(edge, iGraph)];

        m_verticiesInFundamentalCycles.insert(in);
        m_verticiesInFundamentalCycles.insert(out);
      }

      //NOTE: Need to remove any 'extra' bits at beginning of stack
      // which may not be part of the cycle
      m_fundamentalCycles.emplace_back(std::move(tempStack));
    }

    void forward_or_cross_edge(Edge iEdge, Graph const& iGraph) {
      typedef typename boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
      IndexMap const& index = get(boost::vertex_index, iGraph);

      if (kRootVertexIndex != index[source(m_stack.front(), iGraph)]) {
        //this part of the graph is not connected to data processing
        return;
      }

      const unsigned int out = index[target(iEdge, iGraph)];

      //If this is a crossing edge whose out vertex is part of a fundamental cycle
      // then this path is also part of a cycle
      if (m_verticiesInFundamentalCycles.end() == m_verticiesInFundamentalCycles.find(out)) {
        return;
      }

      for (auto const& cycle : m_fundamentalCycles) {
        //Is the out vertex in this cycle?
        auto itStartMatch = cycle.end();
        for (auto it = cycle.begin(); it != cycle.end(); ++it) {
          unsigned int inCycle = index[source(*it, iGraph)];

          if (out == inCycle) {
            itStartMatch = it;
            break;
          }
        }
        if (itStartMatch == cycle.end()) {
          //this cycle isn't the one which uses the vertex from the stack
          continue;
        }

        //tempStack will hold a stack that could have been found by depth first
        // search if module to index ordering had been different
        m_stack.push_back(iEdge);
        auto pop_stack = [](std::vector<Edge>* stack) { stack->pop_back(); };
        std::unique_ptr<std::vector<Edge>, decltype(pop_stack)> guard(&m_stack, pop_stack);
        auto tempStack = findMinimumCycle(m_stack, iGraph);

        //the set of 'in' verticies presently in the stack is used to find where an 'out'
        // vertex from the fundamental cycle connects into the present stack
        std::set<unsigned int> verticiesInStack;
        for (auto const& edge : tempStack) {
          verticiesInStack.insert(index[source(edge, iGraph)]);
        }

        //Now find place in the fundamental cycle that attaches to the stack
        // First see if that happens later in the stack
        auto itLastMatch = cycle.end();
        for (auto it = itStartMatch; it != cycle.end(); ++it) {
          unsigned int outCycle = index[target(*it, iGraph)];
          if (verticiesInStack.end() != verticiesInStack.find(outCycle)) {
            itLastMatch = it;
            break;
          }
        }
        if (itLastMatch == cycle.end()) {
          //See if we can find the attachment to the stack earlier in the cycle
          tempStack.insert(tempStack.end(), itStartMatch, cycle.end());
          for (auto it = cycle.begin(); it != itStartMatch; ++it) {
            unsigned int outCycle = index[target(*it, iGraph)];
            if (verticiesInStack.end() != verticiesInStack.find(outCycle)) {
              itLastMatch = it;
              break;
            }
          }
          if (itLastMatch == cycle.end()) {
            //need to use the full cycle
            //NOTE: this should just retest the same cycle but starting
            // from a different position. If everything is correct, then
            // this should also pass so in principal we could return here.
            //However, as long as this isn't a performance problem, having
            // this additional check could catch problems in the algorithm.
            tempStack.insert(tempStack.end(), cycle.begin(), itStartMatch);
          } else {
            tempStack.insert(tempStack.end(), cycle.begin(), itLastMatch + 1);
          }
        } else {
          if ((itStartMatch == cycle.begin()) and (cycle.end() == (itLastMatch + 1))) {
            //This is just the entire cycle starting where we've already started
            // before. Given the cycle was OK before, it would also be OK this time
            return;
          }
          tempStack.insert(tempStack.end(), itStartMatch, itLastMatch + 1);
        }

        tempStack = findMinimumCycle(tempStack, iGraph);
        checkCycleForProblem(tempStack, iGraph);
      }
    }

  private:
    std::string const& pathName(unsigned int iIndex) const { return m_pathNames[iIndex]; }

    std::string const& moduleName(unsigned int iIndex) const {
      auto itFound = m_indexToNames.find(iIndex);
      assert(itFound != m_indexToNames.end());
      return itFound->second;
    }

    void throwOnError(std::vector<Edge> const& iEdges,
                      boost::property_map<Graph, boost::vertex_index_t>::type const& iIndex,
                      Graph const& iGraph) const {
      std::stringstream oStream;
      oStream << "Module run order problem found: \n";
      bool first_edge = true;
      for (auto const& edge : iEdges) {
        unsigned int in = iIndex[source(edge, iGraph)];
        unsigned int out = iIndex[target(edge, iGraph)];

        if (first_edge) {
          first_edge = false;
        } else {
          oStream << ", ";
        }
        oStream << moduleName(in);

        auto iFound = m_edgeToPathMap.find(SimpleEdge(in, out));
        bool pathDependencyOnly = true;
        for (auto dependency : iFound->second) {
          if (dependency == edm::graph::kDataDependencyIndex) {
            pathDependencyOnly = false;
            break;
          }
        }
        if (pathDependencyOnly) {
          oStream << " after " << moduleName(out) << " [path " << pathName(iFound->second[0]) << "]";
        } else {
          oStream << " consumes " << moduleName(out);
        }
      }
      oStream << "\n Running in the threaded framework would lead to indeterminate results."
                 "\n Please change order of modules in mentioned Path(s) to avoid inconsistent module ordering.";

      throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n") << oStream.str() << "\n";
    }

    std::vector<Edge> findMinimumCycle(std::vector<Edge> const& iCycleEdges, Graph const& iGraph) const {
      //Remove unnecessary edges
      // The graph library scans the verticies so we have edges in the list which are
      // not part of the cycle but are associated to a vertex contributes to the cycle.
      // To find these unneeded edges we work backwards on the edge list looking for cases
      // where the 'in' on the previous edge is not the 'out' for the next edge. When this
      // happens we know that there are additional edges for that same 'in' which can be
      // removed.

      typedef typename boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
      IndexMap const& index = get(boost::vertex_index, iGraph);

      std::vector<Edge> reducedEdges;
      reducedEdges.reserve(iCycleEdges.size());
      reducedEdges.push_back(iCycleEdges.back());
      unsigned int lastIn = index[source(iCycleEdges.back(), iGraph)];
      const unsigned int finalVertex = index[target(iCycleEdges.back(), iGraph)];
      for (auto it = iCycleEdges.rbegin() + 1; it != iCycleEdges.rend(); ++it) {
        unsigned int in = index[source(*it, iGraph)];
        unsigned int out = index[target(*it, iGraph)];
        if (lastIn == out) {
          reducedEdges.push_back(*it);
          lastIn = in;
          if (in == finalVertex) {
            break;
          }
        }
      }
      std::reverse(reducedEdges.begin(), reducedEdges.end());

      return reducedEdges;
    }

    void checkCycleForProblem(std::vector<Edge> const& iCycleEdges, Graph const& iGraph) {
      //For a real problem, we need at least one data dependency
      // we already know we originate from a path because all tests
      // require starting from the root node which connects to all paths
      bool hasDataDependency = false;
      //Since we are dealing with a circle, we initialize the 'last' info with the end of the graph
      typedef typename boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
      IndexMap const& index = get(boost::vertex_index, iGraph);

      unsigned int lastIn = index[source(iCycleEdges.back(), iGraph)];
      unsigned int lastOut = index[target(iCycleEdges.back(), iGraph)];
      bool lastEdgeHasDataDepencency = false;

      std::unordered_set<unsigned int> lastPathsSeen;

      //If a data dependency appears to make us jump off a path but that module actually
      // appears on the path that was left, we need to see if we later come back to that
      // path somewhere before that module. If not than it is a false cycle
      std::unordered_multimap<unsigned int, unsigned int> pathToModulesWhichMustAppearLater;
      bool moduleAppearedEarlierInPath = false;

      for (auto dependency : m_edgeToPathMap.find(SimpleEdge(lastIn, lastOut))->second) {
        if (dependency != edm::graph::kDataDependencyIndex) {
          lastPathsSeen.insert(dependency);
        } else {
          lastEdgeHasDataDepencency = true;
        }
      }
      //Need to check that the
      bool minimumInitialPathsSet = false;
      std::unordered_set<unsigned int> initialPaths(lastPathsSeen);
      std::unordered_set<unsigned int> sharedPaths;
      for (auto const& edge : iCycleEdges) {
        unsigned int in = index[source(edge, iGraph)];
        unsigned int out = index[target(edge, iGraph)];

        auto iFound = m_edgeToPathMap.find(SimpleEdge(in, out));
        std::unordered_set<unsigned int> pathsOnEdge;
        bool edgeHasDataDependency = false;
        for (auto dependency : iFound->second) {
          if (dependency == edm::graph::kDataDependencyIndex) {
            //need to count only if this moves us to a new path
            hasDataDependency = true;
            edgeHasDataDependency = true;
          } else {
            pathsOnEdge.insert(dependency);

            auto const& pathIndicies = m_pathIndexToModuleIndexOrder[dependency];
            auto pathToCheckRange = pathToModulesWhichMustAppearLater.equal_range(dependency);
            for (auto it = pathToCheckRange.first; it != pathToCheckRange.second;) {
              auto moduleIDToCheck = it->second;
              if (moduleIDToCheck == in or moduleIDToCheck == out) {
                auto toErase = it;
                ++it;
                pathToModulesWhichMustAppearLater.erase(toErase);
                continue;
              }
              bool alreadyAdvanced = false;
              for (auto pathIndex : pathIndicies) {
                if (pathIndex == out) {
                  //we must have skipped over the module so the earlier worry about the
                  // module being called on the path was wrong
                  auto toErase = it;
                  ++it;
                  alreadyAdvanced = true;
                  pathToModulesWhichMustAppearLater.erase(toErase);
                  break;
                }
                if (pathIndex == moduleIDToCheck) {
                  //module still earlier on the path
                  break;
                }
              }
              if (not alreadyAdvanced) {
                ++it;
              }
            }
          }
        }
        sharedPaths = intersect(pathsOnEdge, lastPathsSeen);
        if (sharedPaths.empty()) {
          minimumInitialPathsSet = true;
          if ((not edgeHasDataDependency) and (not lastEdgeHasDataDepencency) and (not lastPathsSeen.empty())) {
            //If we jumped from one path to another without a data dependency
            // than the cycle is just because two independent modules were
            // scheduled in different arbitrary order on different paths
            return;
          }
          if (edgeHasDataDependency and not lastPathsSeen.empty()) {
            //If the paths we were on had this module we are going to earlier
            // on their paths than we do not have a real cycle
            bool atLeastOnePathFailed = false;
            std::vector<unsigned int> pathsToWatch;
            pathsToWatch.reserve(lastPathsSeen.size());
            for (auto seenPath : lastPathsSeen) {
              if (pathsOnEdge.end() == pathsOnEdge.find(seenPath)) {
                //we left this path so we now need to see if the module 'out'
                // is on this path ahead of the module 'in'
                bool foundOut = false;
                for (auto seenPathIndex : m_pathIndexToModuleIndexOrder[seenPath]) {
                  if (seenPathIndex == out) {
                    foundOut = true;
                    pathsToWatch.push_back(seenPath);
                  }
                  if (seenPathIndex == lastOut) {
                    if (not foundOut) {
                      atLeastOnePathFailed = true;
                    }
                    break;
                  }
                  if (atLeastOnePathFailed) {
                    break;
                  }
                }
              }
            }
            //If all the paths have the module earlier in their paths
            // then there was no need to jump between paths to get it
            // and this breaks the data cycle
            if (not atLeastOnePathFailed) {
              moduleAppearedEarlierInPath = true;
              for (auto p : pathsToWatch) {
                pathToModulesWhichMustAppearLater.emplace(p, out);
              }
            }
          }
          lastPathsSeen = pathsOnEdge;
        } else {
          lastPathsSeen = sharedPaths;
          if (not minimumInitialPathsSet) {
            initialPaths = sharedPaths;
          }
        }
        lastOut = out;
        lastEdgeHasDataDepencency = edgeHasDataDependency;
      }
      if (moduleAppearedEarlierInPath and not pathToModulesWhichMustAppearLater.empty()) {
        return;
      }
      if (not hasDataDependency) {
        return;
      }
      if ((not initialPaths.empty()) and intersect(initialPaths, sharedPaths).empty()) {
        //The effective start and end paths for the first graph
        // node do not match. This can happen if the node
        // appears on multiple paths
        return;
      }
      throwOnError(iCycleEdges, index, iGraph);
    }

    EdgeToPathMap const& m_edgeToPathMap;
    std::vector<std::vector<unsigned int>> const& m_pathIndexToModuleIndexOrder;
    std::vector<std::string> const& m_pathNames;
    std::unordered_map<unsigned int, std::string> m_indexToNames;
    std::unordered_map<unsigned int, std::vector<unsigned int>> m_pathToModuleIndex;

    std::vector<Edge> m_stack;
    std::vector<std::vector<Edge>> m_fundamentalCycles;
    std::set<unsigned int> m_verticiesInFundamentalCycles;
  };
}  // namespace

void edm::graph::throwIfImproperDependencies(EdgeToPathMap const& iEdgeToPathMap,
                                             std::vector<std::vector<unsigned int>> const& iPathIndexToModuleIndexOrder,
                                             std::vector<std::string> const& iPathNames,
                                             std::unordered_map<unsigned int, std::string> const& iModuleIndexToNames) {
  //Now use boost graph library to find cycles in the dependencies
  std::vector<SimpleEdge> outList;
  outList.reserve(iEdgeToPathMap.size());
  for (auto const& edgeInfo : iEdgeToPathMap) {
    outList.push_back(edgeInfo.first);
  }

  Graph g(outList.begin(), outList.end(), iModuleIndexToNames.size());

  cycle_detector detector(iEdgeToPathMap, iPathIndexToModuleIndexOrder, iPathNames, iModuleIndexToNames);
  boost::depth_first_search(g, boost::visitor(detector));
}
