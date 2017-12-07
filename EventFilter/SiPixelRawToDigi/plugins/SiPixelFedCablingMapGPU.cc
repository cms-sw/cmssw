// Sushil Dubey, Shashi Dugad, TIFR, December 2017
#include <iostream>
#include <algorithm>
#include <set>
#include <map>
#include <vector>
#include <iomanip>

using namespace std;

#include "SiPixelFedCablingMapGPU.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

SiPixelFedCablingMapGPU::SiPixelFedCablingMapGPU(edm::ESTransientHandle<SiPixelFedCablingMap> cablingMap) {
  fedIds = cablingMap->fedIds();
  cabling_ = cablingMap->cablingTree();
}

void SiPixelFedCablingMapGPU::process(CablingMap* &cablingMapGPU, const SiPixelQuality* badPixelInfo, std::set<unsigned int> modules)  {
  int MAX_SIZE = MAX_FED * MAX_LINK * MAX_ROC;
  unsigned int *fedMap   = new unsigned int[MAX_SIZE];
  unsigned int *linkMap  = new unsigned int[MAX_SIZE];
  unsigned int *rocMap   = new unsigned int[MAX_SIZE];
  unsigned int *RawId    = new unsigned int[MAX_SIZE];
  unsigned int *rocInDet = new unsigned int[MAX_SIZE];
  unsigned int *moduleId = new unsigned int[MAX_SIZE];
  bool *badRocs  = new bool[MAX_SIZE];
  bool *modToUnp = new bool[MAX_SIZE];

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
        fedMap[index] = fed;
        linkMap[index] = link;
        rocMap[index] = roc;
        if(pixelRoc != nullptr) {
          RawId[index] = pixelRoc->rawId();
          rocInDet[index] = pixelRoc->idInDetUnit();
          rawIdSet.insert(RawId[index]);

            if(badPixelInfo != nullptr){

                cout<< modules.size() <<endl;
                modToUnp[index] = (modules.size() != 0) && (modules.find(pixelRoc->rawId()) == modules.end());
                badRocs[index] = badPixelInfo->IsRocBad(pixelRoc->rawId(), pixelRoc->idInDetUnit());
                
            }
            else{

                modToUnp[index] = false;
                badRocs[index] = true;
                
            }
            
        }
        else { // store some dummy number
          RawId[index] = 9999;
          rocInDet[index] = 9999;
          modToUnp[index] = false;
          badRocs[index] = true;
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
  for(auto it = rawIdSet.begin(); it != rawIdSet.end(); it++) {
    detIdMap.emplace(*it, module);
    module++;
  }

    for(int i = 1; i < index; i++) {
    cablingMapGPU->fed[i] = fedMap[i];
    cablingMapGPU->link[i] = linkMap[i];
    cablingMapGPU->roc[i] = rocMap[i];
    cablingMapGPU->RawId[i] = RawId[i];
    cablingMapGPU->rocInDet[i] = rocInDet[i];
    cablingMapGPU->badRocs[i] = badRocs[i];
    cablingMapGPU->modToUnp[i] = modToUnp[i];

    if(RawId[i] == 9999) {
      cablingMapGPU->moduleId[i] = 9999;
    }
    else {
      if(detIdMap.find(RawId[i]) == detIdMap.end()) {LogDebug("SiPixelFedCablingMapGPU") << " Not found: "<< RawId[i] << endl; break;}
      auto it = detIdMap.find(RawId[i]);
      cablingMapGPU->moduleId[i] = it->second;
    }
    LogDebug("SiPixelFedCablingMapGPU") <<"----------------------------------------------------------------------------"<<std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << setw(20) << cablingMapGPU->fed[i] << setw(20) << cablingMapGPU->link[i] << setw(20) << cablingMapGPU->roc[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << setw(20) << cablingMapGPU->RawId[i] << setw(20) << cablingMapGPU->rocInDet[i] << setw(20) << cablingMapGPU->moduleId[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") << i << setw(20) << cablingMapGPU->badRocs[i] << setw(20) << cablingMapGPU->modToUnp[i] << std::endl;
    LogDebug("SiPixelFedCablingMapGPU") <<"----------------------------------------------------------------------------"<<std::endl;
  }
  //cout << "size: "<< index << endl;
  cablingMapGPU->size = index-1;
    
  delete[] fedMap;
  delete[] linkMap;
  delete[] rocMap;
  delete[] RawId;
  delete[] rocInDet;
  delete[] moduleId;
  delete[] badRocs;
  delete[] modToUnp;
}
