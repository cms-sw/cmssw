/*
 *
 */

#include <cassert>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

// boost optional (used by boost graph) results in some false positives with -Wmaybe-uninitialized
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <boost/graph/depth_first_search.hpp>
#pragma GCC diagnostic pop

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "HLTrigger/Timer/interface/ProcessCallGraph.h"

ProcessCallGraph::ProcessCallGraph() = default;

// adaptor to use range-based for loops with boost::graph edges(...) and vertices(...) functions
template <typename I>
struct iterator_pair_as_a_range : std::pair<I, I> {
public:
  using std::pair<I, I>::pair;

  I begin() { return this->first; }
  I end() { return this->second; }
};

template <typename I>
iterator_pair_as_a_range<I> make_range(std::pair<I, I> p) {
  return iterator_pair_as_a_range<I>(p);
}

// FIXME
//   - check that the Source has not already been added
void ProcessCallGraph::preSourceConstruction(edm::ModuleDescription const& module) {
  // keep track of the Source module id
  source_ = module.id();

  // create graph vertex for the source module
  boost::add_vertex(graph_);
  graph_.m_graph[module.id()] = {module, edm::EDMModuleType::kSource, true};
}

// FIXME
//  - check that the Source has already been added
//  - check that all module ids are valid (e.g. subprocesses are not being added in
//    the wrong order)
void ProcessCallGraph::preBeginJob(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes,
                                   edm::ProcessContext const& context) {
  unsigned int pid = registerProcess(context);

  // work on the full graph (for the main process) or a subgraph (for a subprocess)
  GraphType& graph = context.isSubProcess() ? graph_.create_subgraph() : graph_.root();

  // set the graph name property to the process name
  boost::get_property(graph, boost::graph_name) = context.processName();

  // create graph vertices associated to all modules in the process
  unsigned int size = pathsAndConsumes.largestModuleID() - boost::num_vertices(graph) + 1;
  for (size_t i = 0; i < size; ++i)
    boost::add_vertex(graph);

  // set the vertices properties (use the module id as the global index into the graph)
  std::vector<unsigned int> modules;
  modules.reserve(size);
  for (edm::ModuleDescription const* module : pathsAndConsumes.allModules()) {
    modules.push_back(module->id());
    graph_.m_graph[module->id()] = {*module, edmModuleTypeEnum(*module), false};
  }

  // add graph edges associated to module dependencies
  for (edm::ModuleDescription const* consumer : pathsAndConsumes.allModules()) {
    for (edm::ModuleDescription const* module : pathsAndConsumes.modulesWhoseProductsAreConsumedBy(consumer->id())) {
      // module `consumer' depends on module `module'
      boost::add_edge(consumer->id(), module->id(), graph_);
    }
  }

  // extract path names from the TriggerNamesService
  edm::service::TriggerNamesService const& tns = *edm::Service<edm::service::TriggerNamesService>();

  // extract the details of the paths and endpaths: name, modules on the path, and their dependencies
  size = pathsAndConsumes.paths().size();
  assert(tns.getTrigPaths().size() == size);
  std::vector<PathType> paths;
  paths.reserve(size);
  for (unsigned int i = 0; i < size; ++i) {
    std::vector<unsigned int> modules;
    for (edm::ModuleDescription const* module : pathsAndConsumes.modulesOnPath(i)) {
      modules.push_back(module->id());
      // mark the modules in the Paths as scheduled
      graph_.m_graph[module->id()].scheduled_ = true;
    }
    auto deps = dependencies(modules);
    paths.emplace_back(tns.getTrigPath(i), modules, deps.first, deps.second);
  }
  size = pathsAndConsumes.endPaths().size();
  std::vector<PathType> endPaths;
  endPaths.reserve(size);
  for (unsigned int i = 0; i < size; ++i) {
    std::vector<unsigned int> modules;
    for (edm::ModuleDescription const* module : pathsAndConsumes.modulesOnEndPath(i)) {
      modules.push_back(module->id());
      // mark the modules in the EndPaths as scheduled
      graph_.m_graph[module->id()].scheduled_ = true;
    }
    auto deps = dependencies(modules);
    endPaths.emplace_back(tns.getEndPath(i), modules, deps.first, deps.second);
  }

  // store the description of process, modules and paths
  process_description_.emplace_back(context.processName(), graph, modules, paths, endPaths);
  assert(process_description_.size() == pid + 1);

  // attach a subprocess to its parent
  if (context.isSubProcess()) {
    unsigned int parent_pid = processId(context.parentProcessContext());
    process_description_[parent_pid].subprocesses_.push_back(pid);
  }
}

// number of modules stored in the call graph
unsigned int ProcessCallGraph::size() const { return boost::num_vertices(graph_); }

// retrieve the ModuleDescriptio associated to the given id and vertex
edm::ModuleDescription const& ProcessCallGraph::source() const { return graph_.m_graph[source_].module_; }

// retrieve the ModuleDescription associated to the given id and vertex
edm::ModuleDescription const& ProcessCallGraph::module(unsigned int module) const {
  return graph_.m_graph[module].module_;
}

// retrieve the full information for a given module
ProcessCallGraph::NodeType const& ProcessCallGraph::operator[](unsigned int module) const {
  return graph_.m_graph[module];
}

// find the dependencies of the given module
std::vector<unsigned int> ProcessCallGraph::depends(unsigned int module) const {
  std::vector<unsigned int> colors(boost::num_vertices(graph_));
  auto colormap = boost::make_container_vertex_map(colors);

  // depht-first visit all vertices starting from the given module
  boost::default_dfs_visitor visitor;
  boost::depth_first_visit(graph_, module, visitor, colormap);

  // count the visited vertices (the `black' ones) in order to properly size the
  // output vector; then fill the dependencies with the list of visited nodes
  unsigned int size = 0;
  for (unsigned int color : colors)
    if (boost::black_color == color)
      ++size;
  std::vector<unsigned int> dependencies(size);
  unsigned j = 0;
  for (unsigned int i = 0; i < colors.size(); ++i)
    if (boost::black_color == colors[i])
      dependencies[j++] = i;
  assert(size == j);

  return dependencies;
}

// find the dependencies of all modules in the given path
//
// return two vector:
//   - the first lists all the dependencies for the whole path
//   - the second lists the one-after-the-last dependency index into the first vector for each module
std::pair<std::vector<unsigned int>, std::vector<unsigned int>> ProcessCallGraph::dependencies(
    std::vector<unsigned int> const& path) {
  std::vector<unsigned int> colors(boost::num_vertices(graph_));
  auto colormap = boost::make_container_vertex_map(colors);

  // first, find and count all the path's modules' dependencies
  boost::default_dfs_visitor visitor;
  for (unsigned int module : path)
    boost::depth_first_visit(graph_, module, visitor, colormap);

  unsigned int size = 0;
  for (unsigned int color : colors)
    if (color == 0)
      ++size;

  // allocate the output vectors
  std::vector<unsigned int> dependencies(size);
  dependencies.resize(0);
  std::vector<unsigned int> indices(path.size());
  indices.resize(0);

  // reset the color map
  for (unsigned int& color : colors)
    color = 0;

  // find again all the dependencies, and record those associated to each module
  struct record_vertices : boost::default_dfs_visitor {
    record_vertices(std::vector<unsigned int>& vertices) : vertices_(vertices) {}

    void discover_vertex(unsigned int vertex, GraphType const& graph) { vertices_.push_back(vertex); }

    std::vector<unsigned int>& vertices_;
  };
  record_vertices recorder(dependencies);

  for (unsigned int module : path) {
    // skip modules that have already been added as dependencies
    if (colors[module] != boost::black_color)
      boost::depth_first_visit(graph_, module, recorder, colormap);
    indices.push_back(dependencies.size());
  }

  return std::make_pair(dependencies, indices);
}

// register a (sub)process and assigns it a "process id"
// if called with a duplicate process name, returns the original process id
unsigned int ProcessCallGraph::registerProcess(edm::ProcessContext const& context) {
  static unsigned int s_id = 0;

  // registerProcess (called by preBeginJob) must be called for the parent process before its subprocess(es)
  if (context.isSubProcess() and process_id_.find(context.parentProcessContext().processName()) == process_id_.end()) {
    throw edm::Exception(edm::errors::LogicError)
        << "ProcessCallGraph::preBeginJob(): called for subprocess \"" << context.processName() << "\""
        << " before being called for its parent process \"" << context.parentProcessContext().processName() << "\"";
  }

  // registerProcess (called by preBeginJob) should be called once or each (sub)process
  auto id = process_id_.find(context.processName());
  if (id != process_id_.end()) {
    throw edm::Exception(edm::errors::LogicError)
        << "ProcessCallGraph::preBeginJob(): called twice for the same "
        << (context.isSubProcess() ? "subprocess" : "process") << " " << context.processName();
  }

  std::tie(id, std::ignore) = process_id_.insert(std::make_pair(context.processName(), s_id++));
  return id->second;
}

// retrieve the "process id" of a process, given its ProcessContex
// throws an exception if the (sub)process was not registered
unsigned int ProcessCallGraph::processId(edm::ProcessContext const& context) const {
  auto id = process_id_.find(context.processName());
  if (id == process_id_.end())
    throw edm::Exception(edm::errors::LogicError)
        << "ProcessCallGraph::processId(): unexpected " << (context.isSubProcess() ? "subprocess" : "process") << " "
        << context.processName();
  return id->second;
}

// retrieve the "process id" of a process, given its ProcessContex
// throws an exception if the (sub)process was not registered
unsigned int ProcessCallGraph::processId(std::string const& processName) const {
  auto id = process_id_.find(processName);
  if (id == process_id_.end())
    throw edm::Exception(edm::errors::LogicError)
        << "ProcessCallGraph::processId(): unexpected (sub)process " << processName;
  return id->second;
}

// retrieve the number of processes
std::vector<ProcessCallGraph::ProcessType> const& ProcessCallGraph::processes() const { return process_description_; }

// retrieve information about a process, given its "process id"
ProcessCallGraph::ProcessType const& ProcessCallGraph::processDescription(unsigned int pid) const {
  return process_description_.at(pid);
}

// retrieve information about a process, given its ProcessContex
ProcessCallGraph::ProcessType const& ProcessCallGraph::processDescription(edm::ProcessContext const& context) const {
  unsigned int pid = processId(context);
  return process_description_[pid];
}

// retrieve information about a process, given its ProcessContex
ProcessCallGraph::ProcessType const& ProcessCallGraph::processDescription(std::string const& processName) const {
  unsigned int pid = processId(processName);
  return process_description_[pid];
}
