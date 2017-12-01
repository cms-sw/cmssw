// Sushil Dubey, Shashi Dugad, TIFR, December 2017
#include <iostream>
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <iomanip>

using namespace std;

#include "SiPixelFedCablingMapGPU.h"

SiPixelFedCablingMapGPU::SiPixelFedCablingMapGPU(edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap) {
  fedIds = cablingMap->fedIds();
  cabling_ = cablingMap->cablingTree();
}

void SiPixelFedCablingMapGPU::process(CablingMap* &cablingMapGPU)  {
  int MAX_SIZE = MAX_FED * MAX_LINK * MAX_ROC;
  unsigned int *RawId    = new unsigned int[MAX_SIZE];
  unsigned int *rocInDet = new unsigned int[MAX_SIZE];
  unsigned int *moduleId = new unsigned int[MAX_SIZE];

  std::set<unsigned int> rawIdSet;

  unsigned int startFed = *(fedIds.begin());
  unsigned int endFed   = *(fedIds.end() - 1);

  sipixelobjects::CablingPathToDetUnit path;
  //sipixelobjects::PixelROC* pixelRoc = nullptr;
  int index = 1;

  for(unsigned int fed = startFed; fed <= endFed; fed++) {
    for(unsigned int link = 1; link <= MAX_LINK; link++) {
      for(unsigned int roc = 1; roc <= MAX_ROC; roc++) {
        path = {fed, link, roc};
        const sipixelobjects::PixelROC* pixelRoc = cabling_->findItem(path);
        if(pixelRoc != nullptr) {
          RawId[index] = pixelRoc->rawId();
          rocInDet[index] = pixelRoc->idInDetUnit();
          rawIdSet.insert(RawId[index]);
        }
        else { // store some dummy number
          RawId[index] = 9999;
          rocInDet[index] = 9999;
        }
        index++;
      }
    }
  } // end of fed loop

//Given FedId, Link and idinLnk; use the following formula
//to get the RawId and idinDU
//index = (FedID-1200) * MAX_LINK* MAX_ROC + (Link-1)* MAX_ROC + idinLnk;
//where, MAX_LINK = 48, MAX_ROC = 8 for Phase1 as mentioned Danek's email
//FedID varies between 1200 to 1338 (In total 108 FED's)
//Link varies between 1 to 48
//idinLnk varies between 1 to 8
  std::map<unsigned int, unsigned int> detIdMap;
  int  module = 0;
  for(auto it = rawIdSet.begin(); it !=rawIdSet.end(); it++) {
    detIdMap.emplace(*it, module);
    module++;
  }

  for(int i = 1; i < index; i++) {
    cablingMapGPU->RawId[i] = RawId[i];
    cablingMapGPU->rocInDet[i] = rocInDet[i];
    if(RawId[i] == 9999) {
      cablingMapGPU->moduleId[i] = 9999;
    }
    else {
      if(detIdMap.find(RawId[i]) == detIdMap.end()) {cout << " Not found: "<< RawId[i] << endl; break;}
      auto it = detIdMap.find(RawId[i]);
      cablingMapGPU->moduleId[i] = it->second;
    }
    //std::cout << cablingMapGPU->RawId[i] << setw(6) << cablingMapGPU->rocInDet[i] << setw(6) << cablingMapGPU->moduleId[i] << std::endl;
  }
  //cout << "size: "<< index << endl;
  cablingMapGPU->size = index-1;
  delete[] RawId;
  delete[] rocInDet;
  delete[] moduleId;
}
