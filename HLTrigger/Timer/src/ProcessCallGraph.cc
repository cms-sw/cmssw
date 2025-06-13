#include <cassert>
#include <string>
#include <vector>

// boost optional (used by boost graph) results in some false positives with -Wmaybe-uninitialized
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <boost/graph/depth_first_search.hpp>
#pragma GCC diagnostic pop

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/Timer/interface/ProcessCallGraph.h"

void ProcessCallGraph::preSourceConstruction(edm::ModuleDescription const& module) {
  // check that the Source has not already been added
  assert(source_ == edm::ModuleDescription::invalidID());

  // keep track of the Source module id
  source_ = module.id();

  // create graph vertex for the source module
  boost::add_vertex(graph_);
  graph_.m_graph[module.id()] = {module, edm::EDMModuleType::kSource, true};
}

void ProcessCallGraph::lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes,
                                                    edm::ProcessContext const& context) {
  // check that the Source has already been added
  assert(source_ != edm::ModuleDescription::invalidID());

  // work on the full graph (for the main process)
  GraphType& graph = graph_.root();

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
  process_description_ = std::make_unique<ProcessType>(
      context.processName(), graph, std::move(modules), std::move(paths), std::move(endPaths));
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

  std::vector<unsigned int> dependencies;
  dependencies.reserve(size);
  std::vector<unsigned int> indices;
  indices.reserve(path.size());

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

ProcessCallGraph::ProcessType const& ProcessCallGraph::processDescription() const { return *process_description_; }
