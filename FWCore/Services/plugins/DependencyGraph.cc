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
  void postBeginJob();

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
    std::string   name;
    unsigned int  id;
    EDMModuleType type;
    bool          scheduled;
  };

  // directed graph, with `vertex_properties` attached to each vertex and no properties attached to edges
  boost::adjacency_list<
      boost::vecS,
      boost::vecS,
      boost::directedS,
      node,
      boost::no_property
  > m_graph;

  std::string m_filename;
  std::string m_name;
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
  desc.addUntracked<std::string>("fileName", "dependency.gv");
  descriptions.add("DependencyGraph", desc);
}


DependencyGraph::DependencyGraph(ParameterSet const & config, ActivityRegistry & registry) :
  m_filename( config.getUntrackedParameter<std::string>("fileName") )
{
  registry.watchPreBeginJob(this, &DependencyGraph::preBeginJob);
  registry.watchPostBeginJob(this, &DependencyGraph::postBeginJob);
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

  // create graph vertex for the source module only if this is not a subprocess
  if (not context.isSubProcess()) {
    boost::add_vertex(m_graph);
    m_graph[0] = { "source", 0, EDMModuleType::Source, true };
    m_name = context.processName();
  } else if (m_name.empty()) {
    // the Service is only in a Subprocess, which is not supported
    edm::LogError("DependencyGraph")
      << "You have requested an instance of the DependencyGraph Service in the \"" << context.processName() 
      << "\" SubProcess, which is not supported.\nPlease move it to the main process.";
    return;
  }

  // create graph vertices associated to all modules in the process
  auto size = pathsAndConsumes.allModules().size();
  for (size_t i = 0; i < size; ++i)
    boost::add_vertex(m_graph);
  for (edm::ModuleDescription const * module: pathsAndConsumes.allModules())
    m_graph[module->id()] = { module->moduleLabel(), module->id(), edmModuleTypeEnum(*module), false };

  // paths and endpaths
  auto const & paths = pathsAndConsumes.paths();
  auto const & endps = pathsAndConsumes.endPaths();

  // mark scheduled modules
  for (unsigned int i = 0; i < paths.size(); ++i)
    for (edm::ModuleDescription const * module: pathsAndConsumes.modulesOnPath(i))
      m_graph[module->id()].scheduled = true;
  for (unsigned int i = 0; i < endps.size(); ++i)
    for (edm::ModuleDescription const * module: pathsAndConsumes.modulesOnEndPath(i))
      m_graph[module->id()].scheduled = true;

  // add graph edges associated to module dependencies
  for (edm::ModuleDescription const * consumer: pathsAndConsumes.allModules())
    for (edm::ModuleDescription const * module: pathsAndConsumes.modulesWhoseProductsAreConsumedBy(consumer->id()))
      boost::add_edge(consumer->id(), module->id(), m_graph);
}

void
DependencyGraph::postBeginJob() {

  if (m_name.empty())
    // the Service is only in a Subprocess, which is not supported
    return;

  // draw the dependency graph
  std::ofstream out(m_filename);
  boost::write_graphviz(
      out,
      m_graph,
      [&](auto & out, auto const & node) {
        out << "[label=\"" << m_graph[node].name << "\" " <<
               "shape=\"" << shapes[static_cast<std::underlying_type_t<EDMModuleType>>(m_graph[node].type)] << "\" " <<
               "style=\"filled\" " <<
               "color=\"black\" " <<
               "fillcolor=\"" << (m_graph[node].scheduled ? "white" : "lightgrey") << "\"]"; },
      [&](auto & out, auto const & edge) { },
      [&](auto & out) {
        out << "label=\"process " << m_name << "\"\n" <<
               "labelloc=top\n"; }
  );
  out.close();
}

namespace edm {
namespace service {

inline
bool isProcessWideService(DependencyGraph const *) {
  return true;
}

} // namespace service
} // namespace edm

// define as a framework servie
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
DEFINE_FWK_SERVICE(DependencyGraph);
