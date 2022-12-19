// -*- C++ -*-
//
// Package:     Services
// Class  :     ResourceInformationService
//
// Implementation:

/** \class edm::service::ResourceInformationService

\author W. David Dagenhart, created 29 April, 2022

*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ResourceInformation.h"

#include <string>
#include <vector>

namespace edm {
  namespace service {

    class ResourceInformationService : public ResourceInformation {
    public:
      ResourceInformationService(ParameterSet const&, ActivityRegistry&);

      static void fillDescriptions(ConfigurationDescriptions&);

      std::vector<AcceleratorType> const& acceleratorTypes() const final;
      std::vector<std::string> const& cpuModels() const final;
      std::vector<std::string> const& gpuModels() const final;

      std::string const& nvidiaDriverVersion() const final;
      int cudaDriverVersion() const final;
      int cudaRuntimeVersion() const final;

      // Same as cpuModels except in a single string with models separated by ", "
      std::string const& cpuModelsFormatted() const final;
      double cpuAverageSpeed() const final;

      void initializeAcceleratorTypes(std::vector<std::string> const& selectedAccelerators) final;
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

      std::vector<AcceleratorType> acceleratorTypes_;
      std::vector<std::string> cpuModels_;
      std::vector<std::string> gpuModels_;

      std::string nvidiaDriverVersion_;
      int cudaDriverVersion_ = 0;
      int cudaRuntimeVersion_ = 0;

      std::string cpuModelsFormatted_;
      double cpuAverageSpeed_ = 0;

      bool locked_ = false;
      bool verbose_;
    };

    inline bool isProcessWideService(ResourceInformationService const*) { return true; }

    ResourceInformationService::ResourceInformationService(ParameterSet const& pset, ActivityRegistry& iRegistry)
        : verbose_(pset.getUntrackedParameter<bool>("verbose")) {
      iRegistry.watchPostBeginJob(this, &ResourceInformationService::postBeginJob);
    }

    void ResourceInformationService::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<bool>("verbose", false);
      descriptions.add("ResourceInformationService", desc);
    }

    std::vector<ResourceInformation::AcceleratorType> const& ResourceInformationService::acceleratorTypes() const {
      return acceleratorTypes_;
    }

    std::vector<std::string> const& ResourceInformationService::cpuModels() const { return cpuModels_; }

    std::vector<std::string> const& ResourceInformationService::gpuModels() const { return gpuModels_; }

    std::string const& ResourceInformationService::nvidiaDriverVersion() const { return nvidiaDriverVersion_; }

    int ResourceInformationService::cudaDriverVersion() const { return cudaDriverVersion_; }

    int ResourceInformationService::cudaRuntimeVersion() const { return cudaRuntimeVersion_; }

    std::string const& ResourceInformationService::cpuModelsFormatted() const { return cpuModelsFormatted_; }

    double ResourceInformationService::cpuAverageSpeed() const { return cpuAverageSpeed_; }

    void ResourceInformationService::initializeAcceleratorTypes(std::vector<std::string> const& selectedAccelerators) {
      if (!locked_) {
        for (auto const& selected : selectedAccelerators) {
          // Test if the string begins with "gpu-"
          if (selected.rfind("gpu-", 0) == 0) {
            acceleratorTypes_.push_back(AcceleratorType::GPU);
            break;
          }
        }
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
    }

    void ResourceInformationService::setCudaDriverVersion(int val) {
      throwIfLocked();
      cudaDriverVersion_ = val;
    }

    void ResourceInformationService::setCudaRuntimeVersion(int val) {
      throwIfLocked();
      cudaRuntimeVersion_ = val;
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

        LogAbsolute("ResourceInformation") << "    acceleratorTypes:";
        if (acceleratorTypes().empty()) {
          LogAbsolute("ResourceInformation") << "        None";
        } else {
          for (auto const& iter : acceleratorTypes()) {
            std::string acceleratorTypeString("unknown type");
            if (iter == AcceleratorType::GPU) {
              acceleratorTypeString = std::string("GPU");
            }
            LogAbsolute("ResourceInformation") << "        " << acceleratorTypeString;
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
