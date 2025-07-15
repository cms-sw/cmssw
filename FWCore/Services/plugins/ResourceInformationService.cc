// -*- C++ -*-
//
// Package:     Services
// Class  :     ResourceInformationService
//
// Implementation:

/** \class edm::service::ResourceInformationService

\author W. David Dagenhart, created 29 April, 2022

*/

#include "FWCore/AbstractServices/interface/ResourceInformation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <string>
#include <vector>

namespace edm {
  namespace service {

    class ResourceInformationService : public ResourceInformation {
    public:
      ResourceInformationService(ParameterSet const&, ActivityRegistry&);

      static void fillDescriptions(ConfigurationDescriptions&);

      HardwareResourcesDescription hardwareResourcesDescription() const final;

      std::vector<std::string> const& selectedAccelerators() const final;
      std::vector<std::string> const& cpuModels() const final;
      std::vector<std::string> const& gpuModels() const final;

      bool hasGpuNvidia() const final;

      std::string const& nvidiaDriverVersion() const final;
      int cudaDriverVersion() const final;
      int cudaRuntimeVersion() const final;

      // Same as cpuModels except in a single string with models separated by ", "
      std::string const& cpuModelsFormatted() const final;
      double cpuAverageSpeed() const final;

      void setSelectedAccelerators(std::vector<std::string> const& selectedAccelerators) final;
      void setCPUModels(std::vector<std::string> const&) final;
      void setGPUModels(std::vector<std::string> const&) final;

      void setNvidiaDriverVersion(std::string const&) final;
      void setCudaDriverVersion(int) final;
      void setCudaRuntimeVersion(int) final;

      void setCpuModelsFormatted(std::string const&) final;
      void setCpuAverageSpeed(double) final;

      void postBeginJob();

    private:
      void throwIfLocked() const;

      std::vector<std::string> selectedAccelerators_;
      std::vector<std::string> cpuModels_;
      std::vector<std::string> gpuModels_;

      std::string nvidiaDriverVersion_;
      int cudaDriverVersion_ = 0;
      int cudaRuntimeVersion_ = 0;

      std::string cpuModelsFormatted_;
      double cpuAverageSpeed_ = 0;

      bool hasGpuNvidia_ = false;
      bool locked_ = false;
      bool verbose_;
    };

    ResourceInformationService::ResourceInformationService(ParameterSet const& pset, ActivityRegistry& iRegistry)
        : verbose_(pset.getUntrackedParameter<bool>("verbose")) {
      iRegistry.watchPostBeginJob(this, &ResourceInformationService::postBeginJob);
    }

    void ResourceInformationService::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<bool>("verbose", false);
      descriptions.add("ResourceInformationService", desc);
    }

    HardwareResourcesDescription ResourceInformationService::hardwareResourcesDescription() const {
      // It is important to have this function defined in a plugin
      // library. It expands the CMS_MICRO_ARCH macro, and loading the
      // library via plugin mechanism rather than as a dependence of
      // another library has the best chance to capture the best
      // microarchitecture that scram decided to use

      HardwareResourcesDescription ret;
      ret.microarchitecture = CMS_MICRO_ARCH;  // macro expands to string literal
      ret.cpuModels = cpuModels();
      ret.selectedAccelerators = selectedAccelerators();
      ret.gpuModels = gpuModels();
      return ret;
    }

    std::vector<std::string> const& ResourceInformationService::selectedAccelerators() const {
      return selectedAccelerators_;
    }

    std::vector<std::string> const& ResourceInformationService::cpuModels() const { return cpuModels_; }

    std::vector<std::string> const& ResourceInformationService::gpuModels() const { return gpuModels_; }

    bool ResourceInformationService::hasGpuNvidia() const { return hasGpuNvidia_; }

    std::string const& ResourceInformationService::nvidiaDriverVersion() const { return nvidiaDriverVersion_; }

    int ResourceInformationService::cudaDriverVersion() const { return cudaDriverVersion_; }

    int ResourceInformationService::cudaRuntimeVersion() const { return cudaRuntimeVersion_; }

    std::string const& ResourceInformationService::cpuModelsFormatted() const { return cpuModelsFormatted_; }

    double ResourceInformationService::cpuAverageSpeed() const { return cpuAverageSpeed_; }

    void ResourceInformationService::setSelectedAccelerators(std::vector<std::string> const& selectedAccelerators) {
      if (!locked_) {
        selectedAccelerators_ = selectedAccelerators;
        locked_ = true;
      }
    }

    void ResourceInformationService::setCPUModels(std::vector<std::string> const& val) {
      throwIfLocked();
      cpuModels_ = val;
    }

    void ResourceInformationService::setGPUModels(std::vector<std::string> const& val) {
      throwIfLocked();
      gpuModels_ = val;
    }

    void ResourceInformationService::setNvidiaDriverVersion(std::string const& val) {
      throwIfLocked();
      nvidiaDriverVersion_ = val;
      hasGpuNvidia_ = true;
    }

    void ResourceInformationService::setCudaDriverVersion(int val) {
      throwIfLocked();
      cudaDriverVersion_ = val;
      hasGpuNvidia_ = true;
    }

    void ResourceInformationService::setCudaRuntimeVersion(int val) {
      throwIfLocked();
      cudaRuntimeVersion_ = val;
      hasGpuNvidia_ = true;
    }

    void ResourceInformationService::setCpuModelsFormatted(std::string const& val) {
      throwIfLocked();
      cpuModelsFormatted_ = val;
    }

    void ResourceInformationService::setCpuAverageSpeed(double val) {
      throwIfLocked();
      cpuAverageSpeed_ = val;
    }

    void ResourceInformationService::throwIfLocked() const {
      if (locked_) {
        // Only Services should modify ResourceInformationService. Service construction is run serially.
        // The lock provides thread safety and prevents modules from modifying ResourceInformationService.
        throw edm::Exception(errors::LogicError)
            << "Attempt to modify member data after ResourceInformationService was locked ";
      }
    }

    void ResourceInformationService::postBeginJob() {
      if (verbose_) {
        LogAbsolute("ResourceInformation") << "ResourceInformationService";
        LogAbsolute("ResourceInformation") << "    cpu models:";
        if (cpuModels().empty()) {
          LogAbsolute("ResourceInformation") << "        None";
        } else {
          for (auto const& iter : cpuModels()) {
            LogAbsolute("ResourceInformation") << "        " << iter;
          }
        }
        LogAbsolute("ResourceInformation") << "    gpu models:";
        if (gpuModels().empty()) {
          LogAbsolute("ResourceInformation") << "        None";
        } else {
          for (auto const& iter : gpuModels()) {
            LogAbsolute("ResourceInformation") << "        " << iter;
          }
        }

        LogAbsolute("ResourceInformation") << "    selectedAccelerators:";
        if (selectedAccelerators().empty()) {
          LogAbsolute("ResourceInformation") << "        None";
        } else {
          for (auto const& iter : selectedAccelerators()) {
            LogAbsolute("ResourceInformation") << "        " << iter;
          }
        }
        LogAbsolute("ResourceInformation") << "    nvidiaDriverVersion: " << nvidiaDriverVersion();
        LogAbsolute("ResourceInformation") << "    cudaDriverVersion: " << cudaDriverVersion();
        LogAbsolute("ResourceInformation") << "    cudaRuntimeVersion: " << cudaRuntimeVersion();
        LogAbsolute("ResourceInformation") << "    cpuModelsFormatted: " << cpuModelsFormatted();
        LogAbsolute("ResourceInformation") << "    cpuAverageSpeed: " << cpuAverageSpeed();
      }
    }

  }  // namespace service
}  // namespace edm

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

using edm::service::ResourceInformationService;
using ResourceInformationMaker =
    edm::serviceregistry::AllArgsMaker<edm::ResourceInformation, ResourceInformationService>;
DEFINE_FWK_SERVICE_MAKER(ResourceInformationService, ResourceInformationMaker);
