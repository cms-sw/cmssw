// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// local includes
#include "SiPixelFedCablingMapGPU.h"

void allocateCablingMap(SiPixelFedCablingMapGPU* & cablingMapHost, SiPixelFedCablingMapGPU* & cablingMapDevice) {
  cudaCheck(cudaMallocHost((void**) & cablingMapHost, sizeof(SiPixelFedCablingMapGPU)));
  cudaCheck(cudaMalloc((void**) & cablingMapDevice, sizeof(SiPixelFedCablingMapGPU)));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->fed,      MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->link,     MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->roc,      MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->RawId,    MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->rocInDet, MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->moduleId, MAX_SIZE_BYTE_INT));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->badRocs,  MAX_SIZE_BYTE_BOOL));
  cudaCheck(cudaMalloc((void**) & cablingMapHost->modToUnp, MAX_SIZE_BYTE_BOOL));
  cudaCheck(cudaMemcpy(cablingMapDevice, cablingMapHost, sizeof(SiPixelFedCablingMapGPU), cudaMemcpyDefault));
}

void deallocateCablingMap(SiPixelFedCablingMapGPU* cablingMapHost, SiPixelFedCablingMapGPU* cablingMapDevice) {
  cudaCheck(cudaFree(cablingMapHost->fed));
  cudaCheck(cudaFree(cablingMapHost->link));
  cudaCheck(cudaFree(cablingMapHost->roc));
  cudaCheck(cudaFree(cablingMapHost->RawId));
  cudaCheck(cudaFree(cablingMapHost->rocInDet));
  cudaCheck(cudaFree(cablingMapHost->moduleId));
  cudaCheck(cudaFree(cablingMapHost->modToUnp));
  cudaCheck(cudaFree(cablingMapHost->badRocs));
  cudaCheck(cudaFree(cablingMapDevice));
  cudaCheck(cudaFreeHost(cablingMapHost));
}


void processCablingMap(SiPixelFedCablingMap const& cablingMap,  TrackerGeometry const& trackerGeom,
                       SiPixelFedCablingMapGPU* cablingMapHost, SiPixelFedCablingMapGPU* cablingMapDevice, 
                       const SiPixelQuality* badPixelInfo, std::set<unsigned int> const& modules) {
  std::vector<unsigned int> const& fedIds = cablingMap.fedIds();
  std::unique_ptr<SiPixelFedCablingTree> const& cabling = cablingMap.cablingTree();

  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  fedMap(MAX_SIZE);
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  linkMap(MAX_SIZE);
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  rocMap(MAX_SIZE);
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  RawId(MAX_SIZE);
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  rocInDet(MAX_SIZE);
  std::vector<unsigned int,  CUDAHostAllocator<unsigned int>>  moduleId(MAX_SIZE);
  std::vector<unsigned char, CUDAHostAllocator<unsigned char>> badRocs(MAX_SIZE);
  std::vector<unsigned char, CUDAHostAllocator<unsigned char>> modToUnp(MAX_SIZE);

  unsigned int startFed = *(fedIds.begin());
  unsigned int endFed   = *(fedIds.end() - 1);

  sipixelobjects::CablingPathToDetUnit path;
  int index = 1;

  for (unsigned int fed = startFed; fed <= endFed; fed++) {
    for (unsigned int link = 1; link <= MAX_LINK; link++) {
      for (unsigned int roc = 1; roc <= MAX_ROC; roc++) {
        path = {fed, link, roc};
        const sipixelobjects::PixelROC* pixelRoc = cabling->findItem(path);
        fedMap[index] = fed;
        linkMap[index] = link;
        rocMap[index] = roc;
        if (pixelRoc != nullptr) {
          RawId[index] = pixelRoc->rawId();
          rocInDet[index] = pixelRoc->idInDetUnit();
          modToUnp[index] = (modules.size() != 0) && (modules.find(pixelRoc->rawId()) == modules.end());
          if (badPixelInfo != nullptr)
            badRocs[index] = badPixelInfo->IsRocBad(pixelRoc->rawId(), pixelRoc->idInDetUnit());
          else
            badRocs[index] = false;
        } else { // store some dummy number
          RawId[index] = 9999;
          rocInDet[index] = 9999;
          modToUnp[index] = true;
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

  cudaDeviceSynchronize();

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
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << (bool)badRocs[i] << std::setw(20) << (bool)modToUnp[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << "----------------------------------------------------------------------------" << std::endl;
  }

  cablingMapHost->size = index-1;
  cudaCheck(cudaMemcpy(cablingMapHost->fed,      fedMap.data(),   fedMap.size()   * sizeof(unsigned int),  cudaMemcpyDefault));
  cudaCheck(cudaMemcpy(cablingMapHost->link,     linkMap.data(),  linkMap.size()  * sizeof(unsigned int),  cudaMemcpyDefault));
  cudaCheck(cudaMemcpy(cablingMapHost->roc,      rocMap.data(),   rocMap.size()   * sizeof(unsigned int),  cudaMemcpyDefault));
  cudaCheck(cudaMemcpy(cablingMapHost->RawId,    RawId.data(),    RawId.size()    * sizeof(unsigned int),  cudaMemcpyDefault));
  cudaCheck(cudaMemcpy(cablingMapHost->rocInDet, rocInDet.data(), rocInDet.size() * sizeof(unsigned int),  cudaMemcpyDefault));
  cudaCheck(cudaMemcpy(cablingMapHost->moduleId, moduleId.data(), moduleId.size() * sizeof(unsigned int),  cudaMemcpyDefault));
  cudaCheck(cudaMemcpy(cablingMapHost->badRocs,  badRocs.data(),  badRocs.size()  * sizeof(unsigned char), cudaMemcpyDefault));
  cudaCheck(cudaMemcpy(cablingMapHost->modToUnp, modToUnp.data(), modToUnp.size() * sizeof(unsigned char), cudaMemcpyDefault));
  cudaCheck(cudaMemcpy(cablingMapDevice, cablingMapHost, sizeof(SiPixelFedCablingMapGPU), cudaMemcpyDefault));
  cudaDeviceSynchronize();
}

void
processGainCalibration(SiPixelGainCalibrationForHLT const & gains, TrackerGeometry const& geom, SiPixelGainForHLTonGPU * & gainsOnGPU, SiPixelGainForHLTonGPU::DecodingStructure * & gainDataOnGPU) {
  // bizzarre logic (looking for fist strip-det) don't ask
  auto const & dus = geom.detUnits();
  unsigned m_detectors = dus.size();
  for(unsigned int i=1;i<7;++i) {
    if(geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() &&
        dus[geom.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isTrackerStrip()) {
      if(geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < m_detectors) m_detectors = geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
    }
  }

  /*
  std::cout << "caching calibs for " << m_detectors << " pixel detectors of size " << gains.data().size() << std::endl;
  std::cout << "sizes " << sizeof(char) << ' ' << sizeof(uint8_t) << ' ' << sizeof(SiPixelGainForHLTonGPU::DecodingStructure) << std::endl;
  */

  SiPixelGainForHLTonGPU * gg;
  cudaCheck(cudaMallocHost((void**) & gg, sizeof(SiPixelGainForHLTonGPU)));

  assert(nullptr==gainDataOnGPU);
  cudaCheck(cudaMalloc((void**) & gainDataOnGPU, gains.data().size()));
  cudaCheck(cudaMalloc((void**) & gainsOnGPU, sizeof(SiPixelGainForHLTonGPU)));
  // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
  cudaCheck(cudaMemcpy(gainDataOnGPU, gains.data().data(), gains.data().size(), cudaMemcpyDefault));

  gg->v_pedestals = gainDataOnGPU;

  // do not read back from the (possibly write-combined) memory buffer
  auto minPed  = gains.getPedLow();
  auto maxPed  = gains.getPedHigh();
  auto minGain = gains.getGainLow();
  auto maxGain = gains.getGainHigh();
  auto nBinsToUseForEncoding = 253;

  // we will simplify later (not everything is needed....)
  gg->minPed_ = minPed;
  gg->maxPed_ = maxPed;
  gg->minGain_= minGain;
  gg->maxGain_= maxGain;

  gg->numberOfRowsAveragedOver_ = 80;
  gg->nBinsToUseForEncoding_    = nBinsToUseForEncoding;
  gg->deadFlag_                 = 255;
  gg->noisyFlag_                = 254;

  gg->pedPrecision  = static_cast<float>(maxPed - minPed) / nBinsToUseForEncoding;
  gg->gainPrecision = static_cast<float>(maxGain - minGain) / nBinsToUseForEncoding;

  /*
  std::cout << "precisions g " << gg->pedPrecision << ' ' << gg->gainPrecision << std::endl;
  */

  // fill the index map
  auto const & ind = gains.getIndexes();  
  /*
  std::cout << ind.size() << " " << m_detectors << std::endl;
  */

  for (auto i=0U; i<m_detectors; ++i) {
    auto p = std::lower_bound(ind.begin(),ind.end(),dus[i]->geographicalId().rawId(),SiPixelGainCalibrationForHLT::StrictWeakOrdering());
    assert (p!=ind.end() && p->detid==dus[i]->geographicalId());
    assert(p->iend<=gains.data().size());
    assert(p->iend>=p->ibegin);
    assert(0==p->ibegin%2);
    assert(0==p->iend%2);
    assert(p->ibegin!=p->iend);
    assert(p->ncols>0);
    gg->rangeAndCols[i] = std::make_pair(SiPixelGainForHLTonGPU::Range(p->ibegin,p->iend), p->ncols);
    // if (ind[i].detid!=dus[i]->geographicalId()) std::cout << ind[i].detid<<"!="<<dus[i]->geographicalId() << std::endl;
    // gg->rangeAndCols[i] = std::make_pair(SiPixelGainForHLTonGPU::Range(ind[i].ibegin,ind[i].iend), ind[i].ncols);
  }

  cudaCheck(cudaMemcpy(gainsOnGPU, gg, sizeof(SiPixelGainForHLTonGPU), cudaMemcpyDefault));
  cudaFreeHost(gg);
  cudaDeviceSynchronize();
}
