#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <vector>

#include <cuda_runtime.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "SiPixelFedCablingMapGPU.h"
#include "SiPixelFedCablingMapGPU.h"

void processCablingMap(SiPixelFedCablingMap const& cablingMap, SiPixelFedCablingMapGPU* cablingMapGPU, SiPixelFedCablingMapGPU* cablingMapDevice, const SiPixelQuality* badPixelInfo, std::set<unsigned int> const& modules) {
  std::vector<unsigned int> const& fedIds = cablingMap.fedIds();
  std::unique_ptr<SiPixelFedCablingTree> const& cabling = cablingMap.cablingTree();

  std::vector<unsigned int>  fedMap(MAX_SIZE);
  std::vector<unsigned int>  linkMap(MAX_SIZE);
  std::vector<unsigned int>  rocMap(MAX_SIZE);
  std::vector<unsigned int>  RawId(MAX_SIZE);
  std::vector<unsigned int>  rocInDet(MAX_SIZE);
  std::vector<unsigned int>  moduleId(MAX_SIZE);
  std::vector<unsigned char> badRocs(MAX_SIZE);
  std::vector<unsigned char> modToUnp(MAX_SIZE);
  std::set<unsigned int>     rawIdSet;

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
          rawIdSet.insert(RawId[index]);
          if (badPixelInfo != nullptr){
            modToUnp[index] = (modules.size() != 0) && (modules.find(pixelRoc->rawId()) == modules.end());
            badRocs[index] = badPixelInfo->IsRocBad(pixelRoc->rawId(), pixelRoc->idInDetUnit());
          } else{
            modToUnp[index] = false;
            badRocs[index] = true;
          }
        } else { // store some dummy number
          RawId[index] = 9999;
          rocInDet[index] = 9999;
          modToUnp[index] = false;
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
  std::map<unsigned int, unsigned int> detIdMap;
  int module = 0;
  for (auto it = rawIdSet.begin(); it != rawIdSet.end(); it++) {
    detIdMap.emplace(*it, module);
    module++;
  }

  cudaDeviceSynchronize();
  for (int i = 1; i < index; i++) {
    if (RawId[i] == 9999) {
      moduleId[i] = 9999;
    } else {
      auto it = detIdMap.find(RawId[i]);
      if (it == detIdMap.end()) {
        LogDebug("SiPixelFedCablingMapGPU") << " Not found: " << RawId[i] << std::endl;
        break;
      }
      moduleId[i] = it->second;
    }
    LogDebug("SiPixelFed/CablingMapGPU") << "----------------------------------------------------------------------------" << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << fedMap[i]  << std::setw(20) << linkMap[i]  << std::setw(20) << rocMap[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << RawId[i]   << std::setw(20) << rocInDet[i] << std::setw(20) << moduleId[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << std::setw(20) << badRocs[i] << std::setw(20) << modToUnp[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << "----------------------------------------------------------------------------" << std::endl;
  }

  cablingMapGPU->size = index-1;
  cudaCheck(cudaMemcpy(cablingMapGPU->fed,      fedMap.data(),   fedMap.size()   * sizeof(unsigned int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(cablingMapGPU->link,     linkMap.data(),  linkMap.size()  * sizeof(unsigned int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(cablingMapGPU->roc,      rocMap.data(),   rocMap.size()   * sizeof(unsigned int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(cablingMapGPU->RawId,    RawId.data(),    RawId.size()    * sizeof(unsigned int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(cablingMapGPU->rocInDet, rocInDet.data(), rocInDet.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(cablingMapGPU->moduleId, moduleId.data(), moduleId.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(cablingMapGPU->badRocs,  badRocs.data(),  badRocs.size()  * sizeof(unsigned char), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(cablingMapGPU->modToUnp, modToUnp.data(), modToUnp.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(cablingMapDevice, cablingMapGPU, sizeof(SiPixelFedCablingMapGPU), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
}
