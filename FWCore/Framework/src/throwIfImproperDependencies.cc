// -*- C++ -*-
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
  struct cycle_detector : public boost::dfs_visitor<> {
    
    cycle_detector(EdgeToPathMap const& iEdgeToPathMap,
                   std::vector<std::vector<unsigned int>> const& iPathIndexToModuleIndexOrder,
                   std::vector<std::string> const& iPathNames,
                   std::unordered_map<unsigned int,std::string> const& iModuleIndexToNames):
    m_edgeToPathMap(iEdgeToPathMap),
    m_pathIndexToModuleIndexOrder(iPathIndexToModuleIndexOrder),
    m_pathNames(iPathNames),
    m_indexToNames(iModuleIndexToNames){}
    
    void tree_edge(Edge iEdge, Graph const&) {
      m_stack.push_back(iEdge);
    }
    
    void finish_edge(Edge iEdge, Graph const& iGraph) {
      if(not m_stack.empty()) {
        if (iEdge == m_stack.back()) {
          m_stack.pop_back();
        }
      }
    }
    
    //Called if a cycle happens
    void back_edge(Edge iEdge, Graph const& iGraph) {
      
      typedef typename boost::property_map<Graph, boost::vertex_index_t>::type IndexMap;
      IndexMap const& index = get(boost::vertex_index, iGraph);
      
      unsigned int vertex = index[target(iEdge,iGraph)];
      
      //Find last edge which starts with this vertex
      std::list<Edge>::iterator itFirst = m_stack.begin();
      {
        bool seenVertex = false;
        while(itFirst != m_stack.end()) {
          if(not seenVertex) {
            if(index[source(*itFirst,iGraph)] == vertex) {
              seenVertex = true;
            }
          } else
            if (index[source(*itFirst,iGraph)] != vertex) {
              break;
            }
          ++itFirst;
        }
        if(itFirst != m_stack.begin()) {
          --itFirst;
        }
      }
      //This edge has not been added to the stack yet
      // making a copy allows us to add it in but not worry
      // about removing it at the end of the routine
      std::vector<Edge> tempStack;
      tempStack.reserve(m_stack.size()+1);
      tempStack.insert(tempStack.end(),itFirst,m_stack.end());
      tempStack.emplace_back(iEdge);
      //Remove unnecessary edges
      // The graph library scans the verticies so we have edges in the list which are
      // not part of the cycle but are associated to a vertex contributes to the cycle.
      // To find these unneeded edges we work backwards on the edge list looking for cases
      // where the 'in' on the previous edge is not the 'out' for the next edge. When this
      // happens we know that there are additional edges for that same 'in' which can be
      // removed.
      {
        std::vector<Edge> reducedEdges;
        reducedEdges.reserve(tempStack.size());
        reducedEdges.push_back(tempStack.back());
        unsigned int lastIn = index[source(tempStack.back(),iGraph)];
        const unsigned int finalVertex = index[target(tempStack.back(),iGraph)];
        for( auto it = tempStack.rbegin()+1; it != tempStack.rend(); ++it) {
          unsigned int in =index[source(*it,iGraph)];
          unsigned int out =index[target(*it,iGraph)];
          if(lastIn == out) {
            reducedEdges.push_back(*it);
            lastIn = in;
            if(in == finalVertex) {
              break;
            }
          }
        }
        std::reverse(reducedEdges.begin(), reducedEdges.end());
        tempStack.swap(reducedEdges);
      }

      
      //For a real problem, we need at least one data dependency and
      // one path
      bool hasDataDependency =false;
      bool hasPathDependency = false;
      std::unordered_map<unsigned int, unsigned int> pathToCountOfNonDataDependencies;
      unsigned int nNonDataDependencies = 0;
      //Since we are dealing with a circle, we initialize the 'last' info with the end of the graph
      unsigned int lastIn =index[source(tempStack.back(),iGraph)];
      unsigned int lastOut = index[target(tempStack.back(),iGraph)];

      std::unordered_set<unsigned int> lastPathsSeen;
      for(auto dependency: m_edgeToPathMap.find(SimpleEdge(lastIn,lastOut))->second) {
        if(dependency !=std::numeric_limits<unsigned int>::max()) {
          lastPathsSeen.insert(dependency);
        }
      }
      for(auto const& edge: tempStack) {
        unsigned int in =index[source(edge,iGraph)];
        unsigned int out =index[target(edge,iGraph)];
        
        auto iFound = m_edgeToPathMap.find(SimpleEdge(in,out));
        std::unordered_set<unsigned int> pathsOnEdge;
        bool edgeHasDataDependency = false;
        for(auto dependency : iFound->second) {
          if (dependency == std::numeric_limits<unsigned int>::max()) {
            //need to count only if this moves us to a new path
            hasDataDependency = true;
            edgeHasDataDependency = true;
          } else {
            hasPathDependency = true;
            pathsOnEdge.insert(dependency);
          }
        }
        if((pathsOnEdge != lastPathsSeen) ) {
          if(not edgeHasDataDependency and not lastPathsSeen.empty()) {
            //If we jumped from one path to another without a data dependency
            // than the cycle is just becuase two independent modules were
            // scheduled in different arbirary order on different paths
            return;
          }
          if(edgeHasDataDependency and not lastPathsSeen.empty()) {
          //If the paths we were on had this module we are going to earlier
          // on their paths than we do not have a real cycle
            bool atLeastOnePathFailed = false;
            for(auto seenPath: lastPathsSeen) {
              if(pathsOnEdge.end() == pathsOnEdge.find(seenPath)) {
                //we left this path so we now need to see if the module 'out'
                // is on this path ahead of the module 'in'
                bool foundOut = false;
                for(auto index: m_pathIndexToModuleIndexOrder[seenPath]) {
                  if(index == out) {
                    foundOut = true;
                  }
                  if(index == lastOut) {
                    if(not foundOut) {
                      atLeastOnePathFailed = true;
                    }
                    break;
                  }
                }
                if(atLeastOnePathFailed) {
                  break;
                }
              }
            }
            //If all the paths have the module earlier in their paths
            // then there was no need to jump between paths to get it
            // and this breaks the data cycle
            if(not atLeastOnePathFailed) return;
          }
          lastPathsSeen = pathsOnEdge;
          lastOut = out;
        }
        if(not edgeHasDataDependency) {
          ++nNonDataDependencies;
          for(auto pathIndex : pathsOnEdge) {
            pathToCountOfNonDataDependencies[pathIndex] +=1;
          }
        }
      }
      if(not (hasPathDependency and hasDataDependency)) {
        return;
      }
      throwOnError(tempStack,index,iGraph);
      for(auto const& pathToCount : pathToCountOfNonDataDependencies) {
        //If all the non data dependencies are seen on on path
        // then at least two modules are in the wrong order
        if (pathToCount.second == nNonDataDependencies) {
          throwOnError(tempStack,index,iGraph);
        }
      }
    }
  private:
    std::string const& pathName(unsigned int iIndex) const {
      return m_pathNames[iIndex];
    }
    
    std::string const& moduleName(unsigned int iIndex) const {
      auto itFound =m_indexToNames.find(iIndex);
      assert(itFound != m_indexToNames.end());
      return itFound->second;
    }
    
    void
    throwOnError(std::vector<Edge>const& iEdges,
                 boost::property_map<Graph, boost::vertex_index_t>::type const& iIndex,
                 Graph const& iGraph) const {
      std::stringstream oStream;
      oStream <<"Module run order problem found: \n";
      bool first_edge = true;
      for(auto const& edge: iEdges) {
        unsigned int in =iIndex[source(edge,iGraph)];
        unsigned int out =iIndex[target(edge,iGraph)];
        
        if(first_edge) {
          first_edge = false;
        } else {
          oStream<<", ";
        }
        oStream <<moduleName(in);
        
        auto iFound = m_edgeToPathMap.find(SimpleEdge(in,out));
        bool pathDependencyOnly = true;
        for(auto dependency : iFound->second) {
          if (dependency == std::numeric_limits<unsigned int>::max()) {
            pathDependencyOnly = false;
            break;
          }
        }
        if (pathDependencyOnly) {
          oStream <<" after "<<moduleName(out)<<" [path "<<pathName(iFound->second[0])<<"]";
        } else {
          oStream <<" consumes "<<moduleName(out);
        }
      }
      oStream<<"\n Running in the threaded framework would lead to indeterminate results."
      "\n Please change order of modules in mentioned Path(s) to avoid inconsistent module ordering.";
      
      throw edm::Exception(edm::errors::ScheduleExecutionFailure, "Unrunnable schedule\n")
      << oStream.str() << "\n";
    }
    
    EdgeToPathMap const& m_edgeToPathMap;
    std::vector<std::vector<unsigned int>> const& m_pathIndexToModuleIndexOrder;
    std::vector<std::string> const& m_pathNames;
    std::unordered_map<unsigned int,std::string> m_indexToNames;
    std::unordered_map<unsigned int, std::vector<unsigned int>> m_pathToModuleIndex;
    
    std::list<Edge> m_stack;
  };
}


void
edm::graph::throwIfImproperDependencies(EdgeToPathMap const& iEdgeToPathMap,
                                        std::vector<std::vector<unsigned int>> const& iPathIndexToModuleIndexOrder,
                                        std::vector<std::string> const& iPathNames,
                                        std::unordered_map<unsigned int, std::string> const& iModuleIndexToNames) {

  //Now use boost graph library to find cycles in the dependencies
  std::vector<SimpleEdge> outList;
  outList.reserve(iEdgeToPathMap.size());
  for(auto const& edgeInfo: iEdgeToPathMap) {
    outList.push_back(edgeInfo.first);
  }

  Graph g(outList.begin(),outList.end(), iModuleIndexToNames.size());

  cycle_detector detector(iEdgeToPathMap,iPathIndexToModuleIndexOrder,iPathNames,iModuleIndexToNames);
  boost::depth_first_search(g,boost::visitor(detector));
}
