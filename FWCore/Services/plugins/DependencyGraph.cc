/*
 * Simple Service to make a GraphViz graph of the modules runtime dependencies:
 *   - draw hard dependencies according to the "consumes" dependencies;
 *   - draw soft dependencies to reflect the order of scheduled modue in each path;
 *   - draw SubProcesses in subgraphs.
 *
 * Use GraphViz dot to generate an SVG representation of the dependencies:
 *
 *   dot -v -Tsvg dependency.dot -o dependency.svg
 *
 */

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

// boost optional (used by boost graph) results in some false positives with -Wmaybe-uninitialized
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/lookup_edge.hpp>
#pragma GCC diagnostic pop

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace edm::service;

namespace {
  namespace {

    template <typename T>
    std::unordered_set<T> make_unordered_set(std::vector<T> &&entries) {
      std::unordered_set<T> u;
      for (T &entry : entries)
        u.insert(std::move(entry));
      return u;
    }

  }  // namespace
}  // namespace

class DependencyGraph {
public:
  DependencyGraph(const ParameterSet &, ActivityRegistry &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void preSourceConstruction(ModuleDescription const &);
  void lookupInitializationComplete(PathsAndConsumesOfModulesBase const &, ProcessContext const &);

private:
  bool highlighted(std::string const &module) { return (m_highlightModules.find(module) != m_highlightModules.end()); }

  enum class EDMModuleType { Unknown, Source, ESSource, ESProducer, EDAnalyzer, EDProducer, EDFilter, OutputModule };

  static constexpr const char *module_type_desc[]{
      "Unknown", "Source", "ESSource", "ESProducer", "EDAnalyzer", "EDProducer", "EDFilter", "OutputModule"};

  static constexpr const char *shapes[]{
      "note",      // Unknown
      "oval",      // Source
      "cylinder",  // ESSource
      "cylinder",  // ESProducer
      "oval",      // EDAnalyzer
      "box",       // EDProducer
      "diamond",   // EDFilter
      "oval",      // OutputModule
  };

  static EDMModuleType edmModuleTypeEnum(edm::ModuleDescription const &module);

  static const char *edmModuleType(edm::ModuleDescription const &module);

  struct node {
    std::string label;
    std::string class_;
    unsigned int id;
    EDMModuleType type;
    bool scheduled;
  };

  using GraphvizAttributes = std::map<std::string, std::string>;

  // directed graph, with `node` properties attached to each vertex
  using GraphType = boost::subgraph<boost::adjacency_list<
      // edge list
      boost::vecS,
      // vertex list
      boost::vecS,
      boost::directedS,
      // vertex properties
      boost::property<boost::vertex_attribute_t,
                      GraphvizAttributes,  // Graphviz vertex attributes
                      node>,
      // edge propoerties
      boost::property<boost::edge_index_t,
                      int,  // used internally by boost::subgraph
                      boost::property<boost::edge_attribute_t, GraphvizAttributes>>,  // Graphviz edge attributes
      // graph properties
      boost::property<
          boost::graph_name_t,
          std::string,  // name each boost::subgraph
          boost::property<boost::graph_graph_attribute_t,
                          GraphvizAttributes,  // Graphviz graph attributes
                          boost::property<boost::graph_vertex_attribute_t,
                                          GraphvizAttributes,
                                          boost::property<boost::graph_edge_attribute_t, GraphvizAttributes>>>>>>;
  GraphType m_graph;

  std::string m_filename;
  std::unordered_set<std::string> m_highlightModules;

  bool m_showPathDependencies;
  bool m_initialized;
};

constexpr const char *DependencyGraph::module_type_desc[];

constexpr const char *DependencyGraph::shapes[];

DependencyGraph::EDMModuleType DependencyGraph::edmModuleTypeEnum(edm::ModuleDescription const &module) {
  auto const &registry = *edm::pset::Registry::instance();
  auto const &pset = *registry.getMapped(module.parameterSetID());

  if (not pset.existsAs<std::string>("@module_edm_type"))
    return EDMModuleType::Unknown;

  std::string const &t = pset.getParameter<std::string>("@module_edm_type");
  for (EDMModuleType v : {EDMModuleType::Source,
                          EDMModuleType::ESSource,
                          EDMModuleType::ESProducer,
                          EDMModuleType::EDAnalyzer,
                          EDMModuleType::EDProducer,
                          EDMModuleType::EDFilter,
                          EDMModuleType::OutputModule}) {
    if (t == module_type_desc[static_cast<std::underlying_type_t<EDMModuleType>>(v)])
      return v;
  }
  return EDMModuleType::Unknown;
}

const char *DependencyGraph::edmModuleType(edm::ModuleDescription const &module) {
  return module_type_desc[static_cast<std::underlying_type_t<EDMModuleType>>(edmModuleTypeEnum(module))];
}

void DependencyGraph::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("fileName", "dependency.dot");
  desc.addUntracked<std::vector<std::string>>("highlightModules", {});
  desc.addUntracked<bool>("showPathDependencies", true);
  descriptions.add("DependencyGraph", desc);
}

DependencyGraph::DependencyGraph(ParameterSet const &config, ActivityRegistry &registry)
    : m_filename(config.getUntrackedParameter<std::string>("fileName")),
      m_highlightModules(
          make_unordered_set(config.getUntrackedParameter<std::vector<std::string>>("highlightModules"))),
      m_showPathDependencies(config.getUntrackedParameter<bool>("showPathDependencies")),
      m_initialized(false) {
  registry.watchPreSourceConstruction(this, &DependencyGraph::preSourceConstruction);
  registry.watchLookupInitializationComplete(this, &DependencyGraph::lookupInitializationComplete);
}

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

void DependencyGraph::preSourceConstruction(ModuleDescription const &module) {
  // create graph vertex for the source module and fill its attributes
  boost::add_vertex(m_graph);
  m_graph.m_graph[module.id()] =
      node{module.moduleLabel(), module.moduleName(), module.id(), EDMModuleType::Source, true};
  auto &attributes = boost::get(boost::get(boost::vertex_attribute, m_graph), 0);
  attributes["label"] = module.moduleLabel();
  attributes["tooltip"] = module.moduleName();
  attributes["shape"] = shapes[static_cast<std::underlying_type_t<EDMModuleType>>(EDMModuleType::Source)];
  attributes["style"] = "filled";
  attributes["color"] = "black";
  attributes["fillcolor"] = highlighted(module.moduleLabel()) ? "lightgreen" : "white";
}

void DependencyGraph::lookupInitializationComplete(PathsAndConsumesOfModulesBase const &pathsAndConsumes,
                                                   ProcessContext const &context) {
  // if the Service is not in the main Process do not do anything
  if (context.isSubProcess() and not m_initialized) {
    edm::LogError("DependencyGraph") << "You have requested an instance of the DependencyGraph Service in the \""
                                     << context.processName()
                                     << "\" SubProcess, which is not supported.\nPlease move it to the main process.";
    return;
  }

  if (not context.isSubProcess()) {
    // set the graph name property to the process name
    boost::get_property(m_graph, boost::graph_name) = context.processName();
    boost::get_property(m_graph, boost::graph_graph_attribute)["label"] = "process " + context.processName();
    boost::get_property(m_graph, boost::graph_graph_attribute)["labelloc"] = "top";

    // create graph vertices associated to all modules in the process
    auto size = pathsAndConsumes.largestModuleID() - boost::num_vertices(m_graph) + 1;
    for (size_t i = 0; i < size; ++i)
      boost::add_vertex(m_graph);

    m_initialized = true;
  } else {
    // create a subgraph to match the subprocess
    auto &graph = m_graph.create_subgraph();

    // set the subgraph name property to the subprocess name
    boost::get_property(graph, boost::graph_name) = "cluster" + context.processName();
    boost::get_property(graph, boost::graph_graph_attribute)["label"] = "subprocess " + context.processName();
    boost::get_property(graph, boost::graph_graph_attribute)["labelloc"] = "top";

    // create graph vertices associated to all modules in the subprocess
    auto size = pathsAndConsumes.largestModuleID() - boost::num_vertices(m_graph) + 1;
    for (size_t i = 0; i < size; ++i)
      boost::add_vertex(graph);
  }

  // set the vertices properties (use the module id as the global index into the graph)
  for (edm::ModuleDescription const *module : pathsAndConsumes.allModules()) {
    m_graph.m_graph[module->id()] = {
        module->moduleLabel(), module->moduleName(), module->id(), edmModuleTypeEnum(*module), false};

    auto &attributes = boost::get(boost::get(boost::vertex_attribute, m_graph), module->id());
    attributes["label"] = module->moduleLabel();
    attributes["tooltip"] = module->moduleName();
    attributes["shape"] = shapes[static_cast<std::underlying_type_t<EDMModuleType>>(edmModuleTypeEnum(*module))];
    attributes["style"] = "filled";
    attributes["color"] = "black";
    attributes["fillcolor"] = highlighted(module->moduleLabel()) ? "green" : "lightgrey";
  }

  // paths and endpaths
  auto const &paths = pathsAndConsumes.paths();
  auto const &endps = pathsAndConsumes.endPaths();

  // add graph edges associated to module dependencies
  for (edm::ModuleDescription const *consumer : pathsAndConsumes.allModules()) {
    for (edm::ModuleDescription const *module : pathsAndConsumes.modulesWhoseProductsAreConsumedBy(consumer->id())) {
      edm::LogInfo("DependencyGraph") << "module " << consumer->moduleLabel() << " depends on module "
                                      << module->moduleLabel();
      auto edge_status = boost::add_edge(consumer->id(), module->id(), m_graph);
      // highlight the edge between highlighted nodes
      if (highlighted(module->moduleLabel()) and highlighted(consumer->moduleLabel())) {
        auto const &edge = edge_status.first;
        auto &attributes = boost::get(boost::get(boost::edge_attribute, m_graph), edge);
        attributes["color"] = "darkgreen";
      }
    }
  }

  // save each Path and EndPath as a Graphviz subgraph
  for (unsigned int i = 0; i < paths.size(); ++i) {
    // create a subgraph to match the Path
    auto &graph = m_graph.create_subgraph();

    // set the subgraph name property to the Path name
    boost::get_property(graph, boost::graph_name) = paths[i];
    boost::get_property(graph, boost::graph_graph_attribute)["label"] = "Path " + paths[i];
    boost::get_property(graph, boost::graph_graph_attribute)["labelloc"] = "bottom";

    // add to the subgraph the node corresponding to the scheduled modules on the Path
    for (edm::ModuleDescription const *module : pathsAndConsumes.modulesOnPath(i)) {
      boost::add_vertex(module->id(), graph);
    }
  }
  for (unsigned int i = 0; i < endps.size(); ++i) {
    // create a subgraph to match the EndPath
    auto &graph = m_graph.create_subgraph();

    // set the subgraph name property to the EndPath name
    boost::get_property(graph, boost::graph_name) = endps[i];
    boost::get_property(graph, boost::graph_graph_attribute)["label"] = "EndPath " + endps[i];
    boost::get_property(graph, boost::graph_graph_attribute)["labelloc"] = "bottom";

    // add to the subgraph the node corresponding to the scheduled modules on the EndPath
    for (edm::ModuleDescription const *module : pathsAndConsumes.modulesOnEndPath(i)) {
      boost::add_vertex(module->id(), graph);
    }
  }

  // optionally, add a dependency of the TriggerResults module on the PathStatusInserter modules
  const int size = boost::num_vertices(m_graph);
  int triggerResults = -1;
  bool highlightTriggerResults = false;
  for (int i = 0; i < size; ++i) {
    if (m_graph.m_graph[i].label == "TriggerResults") {
      triggerResults = i;
      highlightTriggerResults = highlighted("TriggerResults");
      break;
    }
  }

  // mark the modules in the paths as scheduled, and add a soft dependency to reflect the order of modules along each path
  edm::ModuleDescription const *previous;
  for (unsigned int i = 0; i < paths.size(); ++i) {
    previous = nullptr;
    for (edm::ModuleDescription const *module : pathsAndConsumes.modulesOnPath(i)) {
      m_graph.m_graph[module->id()].scheduled = true;
      auto &attributes = boost::get(boost::get(boost::vertex_attribute, m_graph), module->id());
      attributes["fillcolor"] = highlighted(module->moduleLabel()) ? "lightgreen" : "white";
      if (previous and m_showPathDependencies) {
        edm::LogInfo("DependencyGraph") << "module " << module->moduleLabel() << " follows module "
                                        << previous->moduleLabel() << " in Path " << paths[i];
        auto edge_status = boost::lookup_edge(module->id(), previous->id(), m_graph);
        bool found = edge_status.second;
        if (not found) {
          edge_status = boost::add_edge(module->id(), previous->id(), m_graph);
          auto const &edge = edge_status.first;
          auto &edgeAttributes = boost::get(boost::get(boost::edge_attribute, m_graph), edge);
          edgeAttributes["style"] = "dashed";
          // highlight the edge between highlighted nodes
          if (highlighted(module->moduleLabel()) and highlighted(previous->moduleLabel()))
            edgeAttributes["color"] = "darkgreen";
        }
      }
      previous = module;
    }
    // previous points to the last scheduled module on the path
    if (previous and m_showPathDependencies) {
      // look for the PathStatusInserter module corresponding to this path
      for (int j = 0; j < size; ++j) {
        if (m_graph.m_graph[j].label == paths[i]) {
          edm::LogInfo("DependencyGraph") << "module " << paths[i] << " implicitly follows module "
                                          << previous->moduleLabel() << " in Path " << paths[i];
          // add an edge from the PathStatusInserter module to the last module scheduled on the path
          auto edge_status = boost::add_edge(j, previous->id(), m_graph);
          auto const &edge = edge_status.first;
          auto &edgeAttributes = boost::get(boost::get(boost::edge_attribute, m_graph), edge);
          edgeAttributes["style"] = "dashed";
          // highlight the edge between highlighted nodes
          bool highlightedPath = highlighted(paths[i]);
          if (highlightedPath and highlighted(previous->moduleLabel()))
            edgeAttributes["color"] = "darkgreen";
          if (triggerResults > 0) {
            // add an edge from the TriggerResults module to the PathStatusInserter module
            auto edge_status = boost::add_edge(triggerResults, j, m_graph);
            auto const &edge = edge_status.first;
            auto &edgeAttributes = boost::get(boost::get(boost::edge_attribute, m_graph), edge);
            edgeAttributes["style"] = "dashed";
            // highlight the edge between highlighted nodes
            if (highlightedPath and highlightTriggerResults)
              edgeAttributes["color"] = "darkgreen";
          }
          break;
        }
      }
    }
  }

  // mark the modules in the endpaths as scheduled, and add a soft dependency to reflect the order of modules along each endpath
  for (unsigned int i = 0; i < endps.size(); ++i) {
    previous = nullptr;
    for (edm::ModuleDescription const *module : pathsAndConsumes.modulesOnEndPath(i)) {
      m_graph.m_graph[module->id()].scheduled = true;
      auto &attributes = boost::get(boost::get(boost::vertex_attribute, m_graph), module->id());
      attributes["fillcolor"] = highlighted(module->moduleLabel()) ? "lightgreen" : "white";
      if (previous and m_showPathDependencies) {
        edm::LogInfo("DependencyGraph") << "module " << module->moduleLabel() << " follows module "
                                        << previous->moduleLabel() << " in EndPath " << i;
        auto edge_status = boost::lookup_edge(module->id(), previous->id(), m_graph);
        bool found = edge_status.second;
        if (not found) {
          edge_status = boost::add_edge(module->id(), previous->id(), m_graph);
          auto const &edge = edge_status.first;
          auto &edgeAttributes = boost::get(boost::get(boost::edge_attribute, m_graph), edge);
          edgeAttributes["style"] = "dashed";
          // highlight the edge between highlighted nodes
          if (highlighted(module->moduleLabel()) and highlighted(previous->moduleLabel()))
            edgeAttributes["color"] = "darkgreen";
        }
      }
      previous = module;
    }
  }

  // remove the nodes corresponding to the modules that have been removed from the process
  for (int i = boost::num_vertices(m_graph) - 1; i > 1; --i) {
    if (m_graph.m_graph[i].label.empty())
      boost::remove_vertex(i, m_graph.m_graph);
  }

  // draw the dependency graph
  std::ofstream out(m_filename);
  boost::write_graphviz(out, m_graph);
  out.close();
}

namespace edm {
  namespace service {

    inline bool isProcessWideService(DependencyGraph const *) { return true; }

  }  // namespace service
}  // namespace edm

// define as a framework servie
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(DependencyGraph);
