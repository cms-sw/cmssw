// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "CUDADataFormats/SiPixelCluster/interface/gpuClusteringConstants.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelROCsStatusAndMappingWrapper.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"

SiPixelROCsStatusAndMappingWrapper::SiPixelROCsStatusAndMappingWrapper(SiPixelFedCablingMap const& cablingMap,
                                                                       TrackerGeometry const& trackerGeom,
                                                                       SiPixelQuality const* badPixelInfo)
    : cablingMap_(&cablingMap), modToUnpDefault(pixelgpudetails::MAX_SIZE), hasQuality_(badPixelInfo != nullptr) {
  cudaCheck(cudaMallocHost(&cablingMapHost, sizeof(SiPixelROCsStatusAndMapping)));

  std::vector<unsigned int> const& fedIds = cablingMap.fedIds();
  std::unique_ptr<SiPixelFedCablingTree> const& cabling = cablingMap.cablingTree();

  unsigned int startFed = *(fedIds.begin());
  unsigned int endFed = *(fedIds.end() - 1);

  sipixelobjects::CablingPathToDetUnit path;
  int index = 1;

  for (unsigned int fed = startFed; fed <= endFed; fed++) {
    for (unsigned int link = 1; link <= pixelgpudetails::MAX_LINK; link++) {
      for (unsigned int roc = 1; roc <= pixelgpudetails::MAX_ROC; roc++) {
        path = {fed, link, roc};
        const sipixelobjects::PixelROC* pixelRoc = cabling->findItem(path);
        cablingMapHost->fed[index] = fed;
        cablingMapHost->link[index] = link;
        cablingMapHost->roc[index] = roc;
        if (pixelRoc != nullptr) {
          cablingMapHost->rawId[index] = pixelRoc->rawId();
          cablingMapHost->rocInDet[index] = pixelRoc->idInDetUnit();
          modToUnpDefault[index] = false;
          if (badPixelInfo != nullptr)
            cablingMapHost->badRocs[index] = badPixelInfo->IsRocBad(pixelRoc->rawId(), pixelRoc->idInDetUnit());
          else
            cablingMapHost->badRocs[index] = false;
        } else {  // store some dummy number
          cablingMapHost->rawId[index] = gpuClustering::invalidModuleId;
          cablingMapHost->rocInDet[index] = gpuClustering::invalidModuleId;
          cablingMapHost->badRocs[index] = true;
          modToUnpDefault[index] = true;
        }
        index++;
      }
    }
  }  // end of FED loop

  // Given FedId, Link and idinLnk; use the following formula
  // to get the rawId and idinDU
  // index = (FedID-1200) * MAX_LINK* MAX_ROC + (Link-1)* MAX_ROC + idinLnk;
  // where, MAX_LINK = 48, MAX_ROC = 8 for Phase1 as mentioned Danek's email
  // FedID varies between 1200 to 1338 (In total 108 FED's)
  // Link varies between 1 to 48
  // idinLnk varies between 1 to 8

  for (int i = 1; i < index; i++) {
    if (cablingMapHost->rawId[i] == gpuClustering::invalidModuleId) {
      cablingMapHost->moduleId[i] = gpuClustering::invalidModuleId;
    } else {
      /*
      std::cout << cablingMapHost->rawId[i] << std::endl;
      */
      auto gdet = trackerGeom.idToDetUnit(cablingMapHost->rawId[i]);
      if (!gdet) {
        LogDebug("SiPixelROCsStatusAndMapping") << " Not found: " << cablingMapHost->rawId[i] << std::endl;
        continue;
      }
      cablingMapHost->moduleId[i] = gdet->index();
    }
    LogDebug("SiPixelROCsStatusAndMapping")
        << "----------------------------------------------------------------------------" << std::endl;
    LogDebug("SiPixelROCsStatusAndMapping")
        << i << std::setw(20) << cablingMapHost->fed[i] << std::setw(20) << cablingMapHost->link[i] << std::setw(20)
        << cablingMapHost->roc[i] << std::endl;
    LogDebug("SiPixelROCsStatusAndMapping")
        << i << std::setw(20) << cablingMapHost->rawId[i] << std::setw(20) << cablingMapHost->rocInDet[i]
        << std::setw(20) << cablingMapHost->moduleId[i] << std::endl;
    LogDebug("SiPixelROCsStatusAndMapping")
        << i << std::setw(20) << (bool)cablingMapHost->badRocs[i] << std::setw(20) << std::endl;
    LogDebug("SiPixelROCsStatusAndMapping")
        << "----------------------------------------------------------------------------" << std::endl;
  }

  cablingMapHost->size = index - 1;
}

SiPixelROCsStatusAndMappingWrapper::~SiPixelROCsStatusAndMappingWrapper() { cudaCheck(cudaFreeHost(cablingMapHost)); }

const SiPixelROCsStatusAndMapping* SiPixelROCsStatusAndMappingWrapper::getGPUProductAsync(
    cudaStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cudaStream_t stream) {
    // allocate
    cudaCheck(cudaMalloc(&data.cablingMapDevice, sizeof(SiPixelROCsStatusAndMapping)));

    // transfer
    cudaCheck(cudaMemcpyAsync(
        data.cablingMapDevice, this->cablingMapHost, sizeof(SiPixelROCsStatusAndMapping), cudaMemcpyDefault, stream));
  });
  return data.cablingMapDevice;
}

const unsigned char* SiPixelROCsStatusAndMappingWrapper::getModToUnpAllAsync(cudaStream_t cudaStream) const {
  const auto& data =
      modToUnp_.dataForCurrentDeviceAsync(cudaStream, [this](ModulesToUnpack& data, cudaStream_t stream) {
        cudaCheck(cudaMalloc((void**)&data.modToUnpDefault, pixelgpudetails::MAX_SIZE_BYTE_BOOL));
        cudaCheck(cudaMemcpyAsync(data.modToUnpDefault,
                                  this->modToUnpDefault.data(),
                                  this->modToUnpDefault.size() * sizeof(unsigned char),
                                  cudaMemcpyDefault,
                                  stream));
      });
  return data.modToUnpDefault;
}

cms::cuda::device::unique_ptr<unsigned char[]> SiPixelROCsStatusAndMappingWrapper::getModToUnpRegionalAsync(
    std::set<unsigned int> const& modules, cudaStream_t cudaStream) const {
  auto modToUnpDevice = cms::cuda::make_device_unique<unsigned char[]>(pixelgpudetails::MAX_SIZE, cudaStream);
  auto modToUnpHost = cms::cuda::make_host_unique<unsigned char[]>(pixelgpudetails::MAX_SIZE, cudaStream);

  std::vector<unsigned int> const& fedIds = cablingMap_->fedIds();
  std::unique_ptr<SiPixelFedCablingTree> const& cabling = cablingMap_->cablingTree();

  unsigned int startFed = *(fedIds.begin());
  unsigned int endFed = *(fedIds.end() - 1);

  sipixelobjects::CablingPathToDetUnit path;
  int index = 1;

  for (unsigned int fed = startFed; fed <= endFed; fed++) {
    for (unsigned int link = 1; link <= pixelgpudetails::MAX_LINK; link++) {
      for (unsigned int roc = 1; roc <= pixelgpudetails::MAX_ROC; roc++) {
        path = {fed, link, roc};
        const sipixelobjects::PixelROC* pixelRoc = cabling->findItem(path);
        if (pixelRoc != nullptr) {
          modToUnpHost[index] = (not modules.empty()) and (modules.find(pixelRoc->rawId()) == modules.end());
        } else {  // store some dummy number
          modToUnpHost[index] = true;
        }
        index++;
      }
    }
  }

  cudaCheck(cudaMemcpyAsync(modToUnpDevice.get(),
                            modToUnpHost.get(),
                            pixelgpudetails::MAX_SIZE * sizeof(unsigned char),
                            cudaMemcpyHostToDevice,
                            cudaStream));
  return modToUnpDevice;
}

SiPixelROCsStatusAndMappingWrapper::GPUData::~GPUData() { cudaCheck(cudaFree(cablingMapDevice)); }

SiPixelROCsStatusAndMappingWrapper::ModulesToUnpack::~ModulesToUnpack() { cudaCheck(cudaFree(modToUnpDefault)); }
