// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMap const& cablingMap,
                                                               TrackerGeometry const& trackerGeom,
                                                               SiPixelQuality const* badPixelInfo)
    : cablingMap_(&cablingMap), modToUnpDefault(pixelgpudetails::MAX_SIZE), hasQuality_(badPixelInfo != nullptr) {
  cudaCheck(cudaMallocHost(&cablingMapHost, sizeof(SiPixelFedCablingMapGPU)));

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
          cablingMapHost->RawId[index] = pixelRoc->rawId();
          cablingMapHost->rocInDet[index] = pixelRoc->idInDetUnit();
          modToUnpDefault[index] = false;
          if (badPixelInfo != nullptr)
            cablingMapHost->badRocs[index] = badPixelInfo->IsRocBad(pixelRoc->rawId(), pixelRoc->idInDetUnit());
          else
            cablingMapHost->badRocs[index] = false;
        } else {  // store some dummy number
          cablingMapHost->RawId[index] = 9999;
          cablingMapHost->rocInDet[index] = 9999;
          cablingMapHost->badRocs[index] = true;
          modToUnpDefault[index] = true;
        }
        index++;
      }
    }
  }  // end of FED loop

  // Given FedId, Link and idinLnk; use the following formula
  // to get the RawId and idinDU
  // index = (FedID-1200) * MAX_LINK* MAX_ROC + (Link-1)* MAX_ROC + idinLnk;
  // where, MAX_LINK = 48, MAX_ROC = 8 for Phase1 as mentioned Danek's email
  // FedID varies between 1200 to 1338 (In total 108 FED's)
  // Link varies between 1 to 48
  // idinLnk varies between 1 to 8

  for (int i = 1; i < index; i++) {
    if (cablingMapHost->RawId[i] == 9999) {
      cablingMapHost->moduleId[i] = 9999;
    } else {
      /*
      std::cout << cablingMapHost->RawId[i] << std::endl;
      */
      auto gdet = trackerGeom.idToDetUnit(cablingMapHost->RawId[i]);
      if (!gdet) {
        LogDebug("SiPixelFedCablingMapGPU") << " Not found: " << cablingMapHost->RawId[i] << std::endl;
        continue;
      }
      cablingMapHost->moduleId[i] = gdet->index();
    }
    LogDebug("SiPixelFedCablingMapGPU")
        << "----------------------------------------------------------------------------" << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << cablingMapHost->fed[i] << std::setw(20)
                                        << cablingMapHost->link[i] << std::setw(20) << cablingMapHost->roc[i]
                                        << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << cablingMapHost->RawId[i] << std::setw(20)
                                        << cablingMapHost->rocInDet[i] << std::setw(20) << cablingMapHost->moduleId[i]
                                        << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << (bool)cablingMapHost->badRocs[i] << std::setw(20)
                                        << std::endl;
    LogDebug("SiPixelFedCablingMapGPU")
        << "----------------------------------------------------------------------------" << std::endl;
  }

  cablingMapHost->size = index - 1;
}

SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() { cudaCheck(cudaFreeHost(cablingMapHost)); }

const SiPixelFedCablingMapGPU* SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(cudaStream_t cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cudaStream_t stream) {
    // allocate
    cudaCheck(cudaMalloc(&data.cablingMapDevice, sizeof(SiPixelFedCablingMapGPU)));

    // transfer
    cudaCheck(cudaMemcpyAsync(
        data.cablingMapDevice, this->cablingMapHost, sizeof(SiPixelFedCablingMapGPU), cudaMemcpyDefault, stream));
  });
  return data.cablingMapDevice;
}

const unsigned char* SiPixelFedCablingMapGPUWrapper::getModToUnpAllAsync(cudaStream_t cudaStream) const {
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

cms::cuda::device::unique_ptr<unsigned char[]> SiPixelFedCablingMapGPUWrapper::getModToUnpRegionalAsync(
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

SiPixelFedCablingMapGPUWrapper::GPUData::~GPUData() { cudaCheck(cudaFree(cablingMapDevice)); }

SiPixelFedCablingMapGPUWrapper::ModulesToUnpack::~ModulesToUnpack() { cudaCheck(cudaFree(modToUnpDefault)); }
