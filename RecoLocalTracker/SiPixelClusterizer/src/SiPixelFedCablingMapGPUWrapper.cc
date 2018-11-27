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
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelFedCablingMapGPUWrapper.h"

SiPixelFedCablingMapGPUWrapper::SiPixelFedCablingMapGPUWrapper(SiPixelFedCablingMap const& cablingMap,
                                                               TrackerGeometry const& trackerGeom,
                                                               SiPixelQuality const *badPixelInfo):
  fedMap(pixelgpudetails::MAX_SIZE), linkMap(pixelgpudetails::MAX_SIZE), rocMap(pixelgpudetails::MAX_SIZE),
  RawId(pixelgpudetails::MAX_SIZE), rocInDet(pixelgpudetails::MAX_SIZE), moduleId(pixelgpudetails::MAX_SIZE),
  badRocs(pixelgpudetails::MAX_SIZE),
  hasQuality_(badPixelInfo != nullptr)
{
  std::vector<unsigned int> const& fedIds = cablingMap.fedIds();
  std::unique_ptr<SiPixelFedCablingTree> const& cabling = cablingMap.cablingTree();

  unsigned int startFed = *(fedIds.begin());
  unsigned int endFed   = *(fedIds.end() - 1);

  sipixelobjects::CablingPathToDetUnit path;
  int index = 1;

  for (unsigned int fed = startFed; fed <= endFed; fed++) {
    for (unsigned int link = 1; link <= pixelgpudetails::MAX_LINK; link++) {
      for (unsigned int roc = 1; roc <= pixelgpudetails::MAX_ROC; roc++) {
        path = {fed, link, roc};
        const sipixelobjects::PixelROC* pixelRoc = cabling->findItem(path);
        fedMap[index] = fed;
        linkMap[index] = link;
        rocMap[index] = roc;
        if (pixelRoc != nullptr) {
          RawId[index] = pixelRoc->rawId();
          rocInDet[index] = pixelRoc->idInDetUnit();
          if (badPixelInfo != nullptr)
            badRocs[index] = badPixelInfo->IsRocBad(pixelRoc->rawId(), pixelRoc->idInDetUnit());
          else
            badRocs[index] = false;
        } else { // store some dummy number
          RawId[index] = 9999;
          rocInDet[index] = 9999;
          badRocs[index] = true;
        }
        index++;
      }
    }
  } // end of FED loop

  // Given FedId, Link and idinLnk; use the following formula
  // to get the RawId and idinDU
  // index = (FedID-1200) * MAX_LINK* MAX_ROC + (Link-1)* MAX_ROC + idinLnk;
  // where, MAX_LINK = 48, MAX_ROC = 8 for Phase1 as mentioned Danek's email
  // FedID varies between 1200 to 1338 (In total 108 FED's)
  // Link varies between 1 to 48
  // idinLnk varies between 1 to 8

  for (int i = 1; i < index; i++) {
    if (RawId[i] == 9999) {
      moduleId[i] = 9999;
    } else {
      /*
      std::cout << RawId[i] << std::endl;
      */
      auto gdet = trackerGeom.idToDetUnit(RawId[i]);
      if (!gdet) {
        LogDebug("SiPixelFedCablingMapGPU") << " Not found: " << RawId[i] << std::endl;
        continue;
      }
      moduleId[i] = gdet->index();
    }
    LogDebug("SiPixelFedCablingMapGPU") << "----------------------------------------------------------------------------" << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << fedMap[i]  << std::setw(20) << linkMap[i]  << std::setw(20) << rocMap[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << RawId[i]   << std::setw(20) << rocInDet[i] << std::setw(20) << moduleId[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << (bool)badRocs[i] << std::setw(20) << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << "----------------------------------------------------------------------------" << std::endl;

  }

  size = index-1;
}


SiPixelFedCablingMapGPUWrapper::~SiPixelFedCablingMapGPUWrapper() {}


const SiPixelFedCablingMapGPU *SiPixelFedCablingMapGPUWrapper::getGPUProductAsync(cuda::stream_t<>& cudaStream) const {
  const auto& data = gpuData_.dataForCurrentDeviceAsync(cudaStream, [this](GPUData& data, cuda::stream_t<>& stream) {
      // allocate
      cudaCheck(cudaMallocHost((void**) & data.cablingMapHost, sizeof(SiPixelFedCablingMapGPU)));
      cudaCheck(cudaMalloc((void**) & data.cablingMapDevice, sizeof(SiPixelFedCablingMapGPU)));
      cudaCheck(cudaMalloc((void**) & data.cablingMapHost->fed,      pixelgpudetails::MAX_SIZE_BYTE_INT));
      cudaCheck(cudaMalloc((void**) & data.cablingMapHost->link,     pixelgpudetails::MAX_SIZE_BYTE_INT));
      cudaCheck(cudaMalloc((void**) & data.cablingMapHost->roc,      pixelgpudetails::MAX_SIZE_BYTE_INT));
      cudaCheck(cudaMalloc((void**) & data.cablingMapHost->RawId,    pixelgpudetails::MAX_SIZE_BYTE_INT));
      cudaCheck(cudaMalloc((void**) & data.cablingMapHost->rocInDet, pixelgpudetails::MAX_SIZE_BYTE_INT));
      cudaCheck(cudaMalloc((void**) & data.cablingMapHost->moduleId, pixelgpudetails::MAX_SIZE_BYTE_INT));
      cudaCheck(cudaMalloc((void**) & data.cablingMapHost->badRocs,  pixelgpudetails::MAX_SIZE_BYTE_BOOL));

      // transfer
      data.cablingMapHost->size = this->size;
      cudaCheck(cudaMemcpyAsync(data.cablingMapHost->fed,      this->fedMap.data(),   this->fedMap.size()   * sizeof(unsigned int),  cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(data.cablingMapHost->link,     this->linkMap.data(),  this->linkMap.size()  * sizeof(unsigned int),  cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(data.cablingMapHost->roc,      this->rocMap.data(),   this->rocMap.size()   * sizeof(unsigned int),  cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(data.cablingMapHost->RawId,    this->RawId.data(),    this->RawId.size()    * sizeof(unsigned int),  cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(data.cablingMapHost->rocInDet, this->rocInDet.data(), this->rocInDet.size() * sizeof(unsigned int),  cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(data.cablingMapHost->moduleId, this->moduleId.data(), this->moduleId.size() * sizeof(unsigned int),  cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(data.cablingMapHost->badRocs,  this->badRocs.data(),  this->badRocs.size()  * sizeof(unsigned char), cudaMemcpyDefault, stream.id()));
      cudaCheck(cudaMemcpyAsync(data.cablingMapDevice, data.cablingMapHost, sizeof(SiPixelFedCablingMapGPU), cudaMemcpyDefault, stream.id()));
    });
  return data.cablingMapDevice;
}

SiPixelFedCablingMapGPUWrapper::ModulesToUnpack::ModulesToUnpack(cuda::stream_t<>& cudaStream)
{
  edm::Service<CUDAService> cs;
  modToUnpDevice = cs->make_device_unique<unsigned char[]>(pixelgpudetails::MAX_SIZE, cudaStream);
  modToUnpHost = cs->make_host_unique<unsigned char[]>(pixelgpudetails::MAX_SIZE, cudaStream);
}

void SiPixelFedCablingMapGPUWrapper::ModulesToUnpack::fillAsync(SiPixelFedCablingMap const& cablingMap, std::set<unsigned int> const& modules, cuda::stream_t<>& cudaStream) {
  std::vector<unsigned int> const& fedIds = cablingMap.fedIds();
  std::unique_ptr<SiPixelFedCablingTree> const& cabling = cablingMap.cablingTree();

  unsigned int startFed = *(fedIds.begin());
  unsigned int endFed   = *(fedIds.end() - 1);

  sipixelobjects::CablingPathToDetUnit path;
  int index = 1;

  for (unsigned int fed = startFed; fed <= endFed; fed++) {
    for (unsigned int link = 1; link <= pixelgpudetails::MAX_LINK; link++) {
      for (unsigned int roc = 1; roc <= pixelgpudetails::MAX_ROC; roc++) {
        path = {fed, link, roc};
        const sipixelobjects::PixelROC* pixelRoc = cabling->findItem(path);
        if (pixelRoc != nullptr) {
          modToUnpHost[index] = (modules.size() != 0) && (modules.find(pixelRoc->rawId()) == modules.end());
        } else { // store some dummy number
          modToUnpHost[index] = true;
        }
        index++;
      }
    }
  }

  cuda::memory::async::copy(modToUnpDevice.get(), modToUnpHost.get(), pixelgpudetails::MAX_SIZE * sizeof(unsigned char), cudaStream.id());
}


SiPixelFedCablingMapGPUWrapper::GPUData::~GPUData() {
  if(cablingMapHost != nullptr) {
    cudaCheck(cudaFree(cablingMapHost->fed));
    cudaCheck(cudaFree(cablingMapHost->link));
    cudaCheck(cudaFree(cablingMapHost->roc));
    cudaCheck(cudaFree(cablingMapHost->RawId));
    cudaCheck(cudaFree(cablingMapHost->rocInDet));
    cudaCheck(cudaFree(cablingMapHost->moduleId));
    cudaCheck(cudaFree(cablingMapHost->badRocs));
    cudaCheck(cudaFreeHost(cablingMapHost));
  }
  cudaCheck(cudaFree(cablingMapDevice));
}
