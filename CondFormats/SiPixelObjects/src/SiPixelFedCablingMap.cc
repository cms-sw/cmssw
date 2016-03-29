#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <iostream>

//#define OLD_RAW2DIGI

using namespace sipixelobjects;

void SiPixelFedCablingMap::initializeRocs() {
  const bool PRINT = false;
  if(PRINT) std::cout<<"SiPixelFedCablingMap::initializeRocs - cabling version "
		     <<theVersion<<" "<<version()<<std::endl;

#ifdef OLD_RAW2DIGI

  if(PRINT) std::cout << "initialize PixelRocs works only for phase0" << std::endl;
  for (auto & v : theMap) v.second.initFrameConversion(); 

#else 

  // Decide if it is phase0 or phase1 based on the first fed, 0-phase0, 1200-phase1
  unsigned int fedId = (theMap.begin())->first.fed; // get the first fed
  if(PRINT) std::cout << "initialize PixelRocs works for phase0 & phase1 1st fed " 
		      <<fedId<< std::endl;

  if(fedId>=1200) { // phase1
    for (auto & v : theMap) v.second.initFrameConversionPhase1(); // works
  } else { // phase0
    for (auto & v : theMap) v.second.initFrameConversion(); // works
  }

  
  if(0) {  // for testing 
  for (Map::iterator im = theMap.begin(); im != theMap.end(); im++) {
    unsigned int fedId = im->first.fed;
    unsigned int linkId = im->first.link;
    unsigned int rocId = im->first.roc;
    auto rawDetID = im->second.rawId();
    auto idInDetUnit = im->second.idInDetUnit();
    auto idInLink = im->second.idInLink();
    if(0) std::cout<<" roc in map "<<fedId<<" "<<linkId<<" "<<rocId<<" "<<rawDetID<<" "
	     <<idInDetUnit<<" "<<idInLink<<std::endl;
    //auto v = *im;
    if(fedId>=1200) {
      //v.second.initFrameConversionPhase1(); //  
      im->second.initFrameConversionPhase1(); // 
    } else {
      im->second.initFrameConversion();
    }
  } //  
  } 
#endif
  if(PRINT) std::cout<<"SiPixelFedCablingMap::initializeRocs - END "<<std::endl;
}


bool SiPixelFedCablingMap::Key::operator < (const Key & other) const 
{
  if (fed < other.fed) return true;
  if (fed > other.fed) return false;

  if (link < other.link) return true;
  if (link > other.link) return false;

  if (roc < other.roc) return true;
  if (roc > other.roc) return false;

  return false;
}

SiPixelFedCablingMap::SiPixelFedCablingMap(const SiPixelFedCablingTree *cab) 
  : theVersion(cab->version())
{

  std::cout << "HERE --- SiPixelFedCablingMap CTOR- NEVER CALLED " << std::endl;

  // Never called  
  std::vector<const PixelFEDCabling *> fedList = cab->fedList();
  for (std::vector<const PixelFEDCabling *>::const_iterator ifed=fedList.begin();
   ifed != fedList.end(); ifed++) {
    unsigned int fed = (**ifed).id();
    unsigned int numLink = (**ifed).numberOfLinks();
    for (unsigned int link=1; link <= numLink; link++) {
      const PixelFEDLink * pLink = (**ifed).link(link); 
      if (pLink==0) continue;
      unsigned int linkId = pLink->id();
      if (linkId != 0 && linkId!= link) 
          std::cout << "PROBLEM WITH LINK NUMBER!!!!" << std::endl;
      unsigned int numberROC = pLink->numberOfROCs(); 

      //std::cout<<"  cabling map "<<fed<<" "<<linkId<<" "<<link<<" "
      //       <<numberROC<<std::endl;

      for (unsigned int roc=1; roc <= numberROC; roc++) {
        const PixelROC * pROC = pLink->roc(roc);
        if (pROC==0) continue;
        if (pROC->idInLink() != roc) 
            std::cout << "PROBLEM WITH ROC NUMBER!!!!" << std::endl;
        Key key = {fed, link, roc}; 
        theMap[key] = (*pROC);
      }
    } 
  } // fed loop   
 
  std::cout<<" SiPixelFedCablingMap CTOR: end"<<std::endl;
}

std::unique_ptr<SiPixelFedCablingTree>  SiPixelFedCablingMap::cablingTree() const {

  std::unique_ptr<SiPixelFedCablingTree>  tree(new SiPixelFedCablingTree(theVersion)); 
  for (Map::const_iterator im = theMap.begin(); im != theMap.end(); im++) {
    const sipixelobjects::PixelROC & roc = im->second;
    unsigned int fedId = im->first.fed;
    unsigned int linkId = im->first.link;
    tree->addItem(fedId, linkId,  roc);
  }
  return tree;
}

std::vector<unsigned int> SiPixelFedCablingMap::fedIds() const {
  std::vector<unsigned int> result;
  for (Map::const_iterator im = theMap.begin(); im != theMap.end(); im++) {
    unsigned int fedId = im->first.fed;
    if (find(result.begin(),result.end(),fedId) == result.end()) result.push_back(fedId);
  }
  return result;
}

const sipixelobjects::PixelROC* SiPixelFedCablingMap::findItem(
    const sipixelobjects::CablingPathToDetUnit & path) const {
  const PixelROC* roc = 0;
  Key key = {path.fed, path.link, path.roc};
  Map::const_iterator inMap = theMap.find(key);
  if (inMap!= theMap.end()) roc = &(inMap->second);
  return roc;
}

std::vector<sipixelobjects::CablingPathToDetUnit> SiPixelFedCablingMap::pathToDetUnit(
      uint32_t rawDetId) const {

  std::vector<sipixelobjects::CablingPathToDetUnit> result;
  for (Map::const_iterator im = theMap.begin(); im != theMap.end(); ++im) {
    if(im->second.rawId()==rawDetId ) {
      CablingPathToDetUnit path = {im->first.fed, im->first.link, im->first.roc};
      result.push_back(path);
    }
  }
  return result;
}

