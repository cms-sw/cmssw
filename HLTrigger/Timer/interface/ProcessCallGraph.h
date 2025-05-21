#ifndef HLTrigger_Timer_interface_ProcessCallGraph_h
#define HLTrigger_Timer_interface_ProcessCallGraph_h

#include <memory>
#include <utility>
#include <vector>
#include <string>

// boost optional (used by boost graph) results in some false positives with -Wmaybe-uninitialized
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/lookup_edge.hpp>
#include <boost/graph/subgraph.hpp>
#pragma GCC diagnostic pop

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistryfwd.h"
#include "HLTrigger/Timer/interface/EDMModuleType.h"

class ProcessCallGraph {
public:
  struct NodeType {
    edm::ModuleDescription module_;
    edm::EDMModuleType type_ = edm::EDMModuleType::kUnknown;
    bool scheduled_ = false;
  };

  // directed graph, with `NodeType` properties attached to each vertex
  using GraphType = boost::subgraph<boost::adjacency_list<
      // edge list
      boost::vecS,
      // vertex list
      boost::vecS,
      boost::directedS,
      // vertex properties
      NodeType,
      // edge propoerties, used internally by boost::subgraph
      boost::property<boost::edge_index_t, int>,
      // graph properties, used to name each boost::subgraph
      boost::property<boost::graph_name_t, std::string>>>;

  // store the details of each path: name, modules on the path, and their dependencies
  struct PathType {
    std::string name_;
    std::vector<unsigned int> modules_on_path_;
    std::vector<unsigned int> modules_and_dependencies_;
    std::vector<unsigned int>
        last_dependency_of_module_;  // one-after-the-last dependency of each module, as indices into modules_and_dependencies_

    PathType() = default;

    PathType(std::string name,
             std::vector<unsigned int> mop,
             std::vector<unsigned int> mad,
             std::vector<unsigned int> ldom)
        : name_(std::move(name)),
          modules_on_path_(std::move(mop)),
          modules_and_dependencies_(std::move(mad)),
          last_dependency_of_module_(std::move(ldom)) {}
  };

  // store the details of the process: name, modules call subgraph, modules, paths and endpaths
  struct ProcessType {
    std::string name_;
    GraphType const &graph_;
    std::vector<unsigned int> modules_;
    std::vector<PathType> paths_;
    std::vector<PathType> endPaths_;

    ProcessType(std::string name,
                GraphType const &graph,
                std::vector<unsigned int> modules,
                std::vector<PathType> paths,
                std::vector<PathType> endPaths)
        : name_(std::move(name)),
          graph_(graph),
          modules_(std::move(modules)),
          paths_(std::move(paths)),
          endPaths_(std::move(endPaths)) {}
  };

  // default c'tor
  ProcessCallGraph() = default;

  // to be called from preSourceConstruction(...)
  void preSourceConstruction(edm::ModuleDescription const &);

  // to be called from lookupInitializationComplete(...)
  void lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const &, edm::ProcessContext const &);

  // number of modules stored in the call graph
  unsigned int size() const;

  // retrieve the ModuleDescription associated to the Source
  edm::ModuleDescription const &source() const;

  // retrieve the ModuleDescription associated to the given id
  edm::ModuleDescription const &module(unsigned int module) const;

  // retrieve the full information for a given module
  NodeType const &operator[](unsigned int module) const;

  // find the dependencies of the given module
  std::vector<unsigned int> depends(unsigned int module) const;

  // find the dependencies of all modules in the given path
  std::pair<std::vector<unsigned int>, std::vector<unsigned int>> dependencies(std::vector<unsigned int> const &path);

  // retrieve information about the process
  ProcessType const &processDescription() const;

private:
  GraphType graph_;

  // module id of the Source
  unsigned int source_ = edm::ModuleDescription::invalidID();

  // description of the process
  std::unique_ptr<ProcessType> process_description_;
};

#endif  // not defined HLTrigger_Timer_interface_ProcessCallGraph_h
