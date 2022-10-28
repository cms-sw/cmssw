#ifndef HLTrigger_Timer_interface_ProcessCallGraph_h
#define HLTrigger_Timer_interface_ProcessCallGraph_h

/*
 *
 */

#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <type_traits>

// boost optional (used by boost graph) results in some false positives with -Wmaybe-uninitialized
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/lookup_edge.hpp>
#include <boost/graph/subgraph.hpp>
#pragma GCC diagnostic pop

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "HLTrigger/Timer/interface/EDMModuleType.h"

class ProcessCallGraph {
public:
  struct NodeType {
    edm::ModuleDescription module_;
    edm::EDMModuleType type_;
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

    PathType(PathType const &other) = default;

    PathType(PathType &&other) = default;

    ~PathType() = default;

    PathType &operator=(PathType const &other) = default;
  };

  // store the details of each process: name, modules call subgraph, modules, paths and endpaths, subprocess pids
  struct ProcessType {
    std::string name_;
    GraphType const &graph_;
    std::vector<unsigned int> modules_;
    std::vector<PathType> paths_;
    std::vector<PathType> endPaths_;
    std::vector<unsigned int> subprocesses_;

    ProcessType() = delete;

    ProcessType(std::string name,
                GraphType const &graph,
                std::vector<unsigned int> modules,
                std::vector<PathType> paths,
                std::vector<PathType> endPaths,
                std::vector<unsigned int> subprocesses = {})
        : name_(std::move(name)),
          graph_(graph),
          modules_(std::move(modules)),
          paths_(std::move(paths)),
          endPaths_(std::move(endPaths)),
          subprocesses_(std::move(subprocesses)) {}

    ProcessType(std::string &&name,
                GraphType const &graph,
                std::vector<unsigned int> &&modules,
                std::vector<PathType> &&paths,
                std::vector<PathType> &&endPaths,
                std::vector<unsigned int> &&subprocesses = {})
        : name_(std::move(name)),
          graph_(graph),
          modules_(std::move(modules)),
          paths_(std::move(paths)),
          endPaths_(std::move(endPaths)),
          subprocesses_(std::move(subprocesses)) {}

    ProcessType(ProcessType const &other) = default;
    ProcessType(ProcessType &&other) = default;

    ProcessType &operator=(ProcessType const &other) = delete;
    ProcessType &operator=(ProcessType &&other) = delete;
  };

public:
  // default c'tor
  ProcessCallGraph() = default;

  // to be called from preSourceConstruction(...)
  void preSourceConstruction(edm::ModuleDescription const &);

  // to be called from preBeginJob(...)
  void preBeginJob(edm::PathsAndConsumesOfModulesBase const &, edm::ProcessContext const &);

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

  // retrieve the "process id" of a process, given its ProcessContex
  unsigned int processId(edm::ProcessContext const &) const;

  // retrieve the "process id" of a process, given its name
  unsigned int processId(std::string const &) const;

  // retrieve the processes
  std::vector<ProcessType> const &processes() const;

  // retrieve information about a process, given its "process id"
  ProcessType const &processDescription(unsigned int) const;

  // retrieve information about a process, given its ProcessContex
  ProcessType const &processDescription(edm::ProcessContext const &) const;

  // retrieve information about a process, given its name
  ProcessType const &processDescription(std::string const &) const;

private:
  // register a (sub)process and assigns it a "process id"
  unsigned int registerProcess(edm::ProcessContext const &);

private:
  GraphType graph_;

  // module id of the Source
  unsigned int source_ = edm::ModuleDescription::invalidID();

  // map each (sub)process name to a "process id"
  std::unordered_map<std::string, unsigned int> process_id_;

  // description of each process
  std::vector<ProcessType> process_description_;
};

#endif  // not defined HLTrigger_Timer_interface_ProcessCallGraph_h
