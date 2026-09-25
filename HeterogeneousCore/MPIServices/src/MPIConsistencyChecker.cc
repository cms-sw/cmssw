// -*- C++ -*-
#include <algorithm>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/MPIServices/interface/MPIConsistencyChecker.h"

MPIConsistencyChecker::MPIConsistencyChecker(edm::ParameterSet const&) {}

void MPIConsistencyChecker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("MPIConsistencyChecker", desc);
  descriptions.setComment("This service records and validates MPI sender and receiver module information.");
}

void MPIConsistencyChecker::required() {
  edm::Service<MPIConsistencyChecker> service;
  if (not service.isAvailable()) {
    throw cms::Exception("Configuration") << R"(The MPIConsistencyChecker is required by this module.
Please add it to the configuration, for example via

process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")
)";
  }
}

void MPIConsistencyChecker::recordMPIModuleInfo(bool is_sender,
                                                std::string const& module_label,
                                                std::string const& upstream_label,
                                                int instance,
                                                std::vector<std::string> const& product_types) {
  std::lock_guard<std::mutex> lock(modules_info_mutex_);
  modules_info_.push_back(MPIModuleInfo{is_sender, instance, module_label, product_types});
  module_upstream_labels_.push_back(upstream_label);
}

void MPIConsistencyChecker::getSerializedMPIModuleInfo(std::vector<char>& buffer, std::string const& origin_name) {
  std::lock_guard<std::mutex> lock(modules_info_mutex_);
  auto const& modules = mpi_paths_mappings_[origin_name];
  TBufferFile buf(TBuffer::kWrite);

  TClass* cls = TClass::GetClass<std::vector<MPIModuleInfo>>();

  if (!cls) {
    throw cms::Exception("MPIConsistencyChecker") << "Failed to get TClass for std::vector<MPIModuleInfo>";
  }

  // We use const_cast here because TBufferFile::Streamer requires a non-const pointer, even though it does not modify the vector.
  cls->Streamer(const_cast<std::vector<MPIModuleInfo>*>(&modules), buf);

  buffer.resize(buf.Length());
  std::memcpy(buffer.data(), buf.Buffer(), buf.Length());
}

void MPIConsistencyChecker::deserializeMPIModuleInfo(std::vector<char> const& buffer,
                                                     std::vector<MPIModuleInfo>& info) {
  // We use const_cast here because TBufferFile::Streamer requires a non-const pointer, even though it does not modify the buffer.
  TBufferFile buf(TBuffer::kRead, buffer.size(), const_cast<char*>(buffer.data()), false);
  TClass* cls = TClass::GetClass<std::vector<MPIModuleInfo>>();

  if (!cls) {
    throw cms::Exception("MPIConsistencyChecker") << "Failed to get TClass for std::vector<MPIModuleInfo>";
  }

  cls->Streamer(&info, buf);
}

void MPIConsistencyChecker::registerMPIPathOrigin(std::string const& origin_name) {
  std::lock_guard<std::mutex> lock(modules_info_mutex_);
  mpi_paths_mappings_.try_emplace(origin_name);
}

void MPIConsistencyChecker::reconstructMPIPaths() {
  std::call_once(paths_reconstructed_flag_, [&]() {
    std::lock_guard<std::mutex> lock(modules_info_mutex_);
    std::unordered_map<std::string, size_t> module_by_label;
    for (size_t i = 0; i < modules_info_.size(); ++i) {
      module_by_label.emplace(modules_info_[i].module_label, i);
    }

    std::vector<std::pair<std::string, size_t>> module_paths(modules_info_.size());
    for (size_t i = 0; i < modules_info_.size(); ++i) {
      std::string origin = module_upstream_labels_[i];
      size_t distance = 0;
      std::vector<bool> visited(modules_info_.size(), false);

      while (not mpi_paths_mappings_.count(origin)) {
        auto module = module_by_label.find(origin);
        if (module == module_by_label.end() || visited[module->second]) {
          throw cms::Exception("MPIConsistencyChecker")
              << "Could not assign MPI module " << modules_info_[i].module_label << " to any MPI path";
        }
        visited[module->second] = true;
        origin = module_upstream_labels_[module->second];
        ++distance;
      }
      module_paths[i] = {origin, distance};
    }

    for (auto& path : mpi_paths_mappings_) {
      std::vector<size_t> indices;
      for (size_t i = 0; i < module_paths.size(); ++i) {
        if (module_paths[i].first == path.first) {
          indices.push_back(i);
        }
      }
      std::stable_sort(indices.begin(), indices.end(), [&](size_t left, size_t right) {
        return module_paths[left].second < module_paths[right].second;
      });
      for (auto index : indices) {
        path.second.push_back(modules_info_[index]);
      }
    }
  });
}

void MPIConsistencyChecker::compareMPIModules(std::vector<MPIModuleInfo> const& other,
                                              std::string const& origin_name,
                                              std::string const& other_process_name,
                                              std::string const& this_process_name) {
  std::lock_guard<std::mutex> lock(modules_info_mutex_);
  auto const& local_modules = mpi_paths_mappings_[origin_name];

  // print the local and remote modules info for debugging
  LogDebug("MPIConsistencyChecker") << "Local MPI modules info: ";
  for (auto const& module : local_modules) {
    LogDebug("MPIConsistencyChecker") << "  is_sender: " << module.is_sender << ", instance: " << module.instance
                                      << ", product_types: ";
    for (auto const& type : module.product_types) {
      LogDebug("MPIConsistencyChecker") << "    " << type;
    }
  }
  LogDebug("MPIConsistencyChecker") << "Remote MPI modules info: ";
  for (auto const& module : other) {
    LogDebug("MPIConsistencyChecker") << "  is_sender: " << module.is_sender << ", instance: " << module.instance
                                      << ", product_types: ";
    for (auto const& type : module.product_types) {
      LogDebug("MPIConsistencyChecker") << "    " << type;
    }
  }

  std::vector<std::string> errors;
  if (local_modules.size() != other.size()) {
    std::ostringstream error;
    error << "Mismatch in number of MPI modules between process " << this_process_name << " and process "
          << other_process_name << ": " << local_modules.size() << " vs " << other.size();
    errors.push_back(error.str());
  }
  for (auto const& local_module : local_modules) {
    auto it = std::find_if(other.begin(), other.end(), [&](MPIModuleInfo const& remote_module) {
      return remote_module.is_sender != local_module.is_sender && remote_module.instance == local_module.instance;
    });
    if (it == other.end()) {
      std::ostringstream error;
      error << "No matching sender/receiver found in process " << other_process_name << " for MPI module instance "
            << local_module.instance << " label " << local_module.module_label << " in process " << this_process_name;
      errors.push_back(error.str());
      continue;
    }
    if (local_module.product_types.size() != it->product_types.size()) {
      std::ostringstream error;
      error << "Mismatch in number of product types between sender module " << local_module.instance << " label "
            << local_module.module_label << " in process " << this_process_name << " and receiver module "
            << it->instance << " in process " << other_process_name << ": " << local_module.product_types.size()
            << " vs " << it->product_types.size();
      errors.push_back(error.str());
      continue;
    }
    for (size_t i = 0; i < local_module.product_types.size(); ++i) {
      if (local_module.product_types[i] != it->product_types[i]) {
        std::ostringstream error;
        error << "Mismatch in product type at index " << i << " between sender module " << local_module.instance
              << " label " << local_module.module_label << " in process " << this_process_name << " ("
              << local_module.product_types[i] << ") and receiver module " << it->instance << " label "
              << it->module_label << " in process " << other_process_name << " (" << it->product_types[i] << ")";
        errors.push_back(error.str());
      }
    }
  }
  if (not errors.empty()) {
    cms::Exception exception("MPIConsistencyChecker");
    exception << "Found " << errors.size() << " MPI consistency error(s):";
    for (auto const& error : errors) {
      exception << "\n  - " << error;
    }
    throw exception;
  }
}
