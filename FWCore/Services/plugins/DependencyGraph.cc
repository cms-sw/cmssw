#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace edm::service;

class DependencyGraph {
public:
  DependencyGraph(const ParameterSet&,ActivityRegistry&);

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  void preBeginJob(PathsAndConsumesOfModulesBase const &, ProcessContext const &);

private:
  enum class EDMModuleType {
    Unknown,
    Source,
    ESSource,
    ESProducer,
    EDAnalyzer,
    EDProducer,
    EDFilter,
    OutputModule
  };

  static constexpr
  const char * module_type_desc[] {
    "Unknown",
    "Source",
    "ESSource",
    "ESProducer",
    "EDAnalyzer",
    "EDProducer",
    "EDFilter",
    "OutputModule"
  };

  static constexpr
  const char * shapes[] {
    "note",     // Unknown
    "oval",     // Source
    "cylinder", // ESSource
    "cylinder", // ESProducer
    "oval",     // EDAnalyzer
    "box",      // EDProducer
    "diamond",  // EDFilter
    "oval",     // OutputModule
  };

  static
  EDMModuleType edmModuleTypeEnum(edm::ModuleDescription const & module);

  static
  const char * edmModuleType(edm::ModuleDescription const & module);

  struct node {
    std::string  name;
    unsigned int id;
    EDMModuleType  type;
    bool         scheduled;
  };

};

constexpr
const char * DependencyGraph::module_type_desc[];

constexpr
const char * DependencyGraph::shapes[];


DependencyGraph::EDMModuleType DependencyGraph::edmModuleTypeEnum(edm::ModuleDescription const & module)
{
  auto const & registry = * edm::pset::Registry::instance();
  auto const & pset = * registry.getMapped(module.parameterSetID());

  if (not pset.existsAs<std::string>("@module_edm_type"))
    return EDMModuleType::Unknown;

  std::string const & t = pset.getParameter<std::string>("@module_edm_type");
  for (EDMModuleType v: {
    EDMModuleType::Source,
    EDMModuleType::ESSource,
    EDMModuleType::ESProducer,
    EDMModuleType::EDAnalyzer,
    EDMModuleType::EDProducer,
    EDMModuleType::EDFilter,
    EDMModuleType::OutputModule
  }) {
    if (t == module_type_desc[static_cast<std::underlying_type_t<EDMModuleType>>(v)])
      return v;
  }
  return EDMModuleType::Unknown;
}


const char * DependencyGraph::edmModuleType(edm::ModuleDescription const & module)
{
  return module_type_desc[static_cast<std::underlying_type_t<EDMModuleType>>(edmModuleTypeEnum(module))];
}


void
DependencyGraph::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  descriptions.add("DependencyGraph", desc);
}


DependencyGraph::DependencyGraph(ParameterSet const & iConfig, ActivityRegistry & iRegistry)
{
  iRegistry.watchPreBeginJob(this, &DependencyGraph::preBeginJob);
}


// adaptor to use range-based for loops with boost::graph edges(...) and vertices(...) functions
template <typename I>
struct iterator_pair_as_a_range : std::pair<I, I>
{
public:
  using std::pair<I, I>::pair;

  I begin() { return this->first; }
  I end()   { return this->second; }
};

template <typename I>
iterator_pair_as_a_range<I> make_range(std::pair<I, I> p)
{
  return iterator_pair_as_a_range<I>(p);
}


void
DependencyGraph::preBeginJob(PathsAndConsumesOfModulesBase const & pathsAndConsumes, ProcessContext const & context) {

  // build a directed graph, with `vertex_properties` attached to each vertex and no properties attached to edges
  boost::adjacency_list<
      boost::vecS,
      boost::vecS,
      boost::directedS,
      node,
      boost::no_property
  > graph;

  // map module id's to graph nodes
  std::map<unsigned int, decltype(boost::add_vertex(graph))> nodes;

  // paths and endpaths
  auto const & paths = pathsAndConsumes.paths();
  auto const & endps = pathsAndConsumes.endPaths();

  // create graph vertices associated to all modules in the process
  for (edm::ModuleDescription const * module: pathsAndConsumes.allModules()) {
    auto v = boost::add_vertex(graph);
    graph[v] = { module->moduleLabel(), module->id(), edmModuleTypeEnum(*module), false };
    nodes[module->id()] = v;
  }

  // mark scheduled modules
  for (unsigned int i = 0; i < paths.size(); ++i)
    for (edm::ModuleDescription const * module: pathsAndConsumes.modulesOnPath(i))
      graph[nodes[module->id()]].scheduled = true;
  for (unsigned int i = 0; i < endps.size(); ++i)
    for (edm::ModuleDescription const * module: pathsAndConsumes.modulesOnEndPath(i))
      graph[nodes[module->id()]].scheduled = true;

  // add graph edges associated to module dependencies
  for (auto vertex: make_range(boost::vertices(graph)))
    for (edm::ModuleDescription const * module: pathsAndConsumes.modulesWhoseProductsAreConsumedBy(graph[vertex].id))
      boost::add_edge(vertex, nodes[module->id()], graph);

  // draw the dependency graph
  std::ofstream out("dependency.gv");
  boost::write_graphviz(
      out,
      graph,
      [&](auto & out, auto const & node) {
        out << "[label=\"" << graph[node].name << "\" " <<
               "shape=\"" << shapes[static_cast<std::underlying_type_t<EDMModuleType>>(graph[node].type)] << "\" " <<
               "style=\"filled\" " <<
               "color=\"black\" " <<
               "fillcolor=\"" << (graph[node].scheduled ? "white" : "lightgrey") << "\"]"; },
      [&](auto & out, auto const & edge) { },
      [&](auto & out) { 
        out << "label=\"process " << context.processName() << "\"\n" <<
               "labelloc=top\n"; }
  );
  out.close();
}


// define as a framework servie
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(DependencyGraph);
