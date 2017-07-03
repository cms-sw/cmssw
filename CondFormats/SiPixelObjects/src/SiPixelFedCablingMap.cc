#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <iostream>

using namespace sipixelobjects;

void SiPixelFedCablingMap::initializeRocs() {

  // OLD Method 
  //for (auto & v : theMap) v.second.initFrameConversion(); 
  // below is the new code, works for phase0 and phase1

  // Decide if it is phase0 or phase1 based on the first fed, 0-phase0, 1200-phase1
  unsigned int fedId = (theMap.begin())->first.fed; // get the first fed

  // Specifically for CMSSW_9_0_X, we need to call a different version of the frame 
  // conversion steered by the version name in the cabling map
  if (theVersion.find("CMSSW_9_0_X")!=std::string::npos) {
    for (auto & v : theMap) v.second.initFrameConversionPhase1_CMSSW_9_0_X(); // works
    std::cout<<"*** Found CMSSW_9_0_X specific cabling map\n";
    return;
  }

  if(fedId>=FEDNumbering::MINSiPixeluTCAFEDID) { // phase1 >= 1200
    for (auto & v : theMap) v.second.initFrameConversionPhase1(); // works
  } else { // phase0
    for (auto & v : theMap) v.second.initFrameConversion(); // works
  }
  
  // if(0) {  // for testing 
  //   for (Map::iterator im = theMap.begin(); im != theMap.end(); im++) {
  //     unsigned int fedId = im->first.fed;
  //     unsigned int linkId = im->first.link;
  //     unsigned int rocId = im->first.roc;
  //     auto rawDetID = im->second.rawId();
  //     auto idInDetUnit = im->second.idInDetUnit();
  //     auto idInLink = im->second.idInLink();
  //     //auto v = *im;
  //     if(fedId>=1200) {
  // 	//v.second.initFrameConversionPhase1(); //  
  // 	im->second.initFrameConversionPhase1(); // 
  //     } else {
  // 	im->second.initFrameConversion();
  //     }
  //   } //  
  // } // end if(0)

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

  // Never called  
  std::vector<const PixelFEDCabling *> fedList = cab->fedList();
  for (std::vector<const PixelFEDCabling *>::const_iterator ifed=fedList.begin();
   ifed != fedList.end(); ifed++) {
    unsigned int fed = (**ifed).id();
    unsigned int numLink = (**ifed).numberOfLinks();
    for (unsigned int link=1; link <= numLink; link++) {
      const PixelFEDLink * pLink = (**ifed).link(link); 
      if (pLink==nullptr) continue;
      //unsigned int linkId = pLink->id();
      //if (linkId != 0 && linkId!= link) 
      //  std::cout << "PROBLEM WITH LINK NUMBER!!!!" << std::endl;
      unsigned int numberROC = pLink->numberOfROCs(); 

      for (unsigned int roc=1; roc <= numberROC; roc++) {
        const PixelROC * pROC = pLink->roc(roc);
        if (pROC==nullptr) continue;
        //if (pROC->idInLink() != roc) 
	//  std::cout << "PROBLEM WITH ROC NUMBER!!!!" << std::endl;
        Key key = {fed, link, roc}; 
        theMap[key] = (*pROC);
      }
    } 
  } // fed loop   
 
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
  const PixelROC* roc = nullptr;
  Key key = {path.fed, path.link, path.roc};
  Map::const_iterator inMap = theMap.find(key);
  if (inMap!= theMap.end()) roc = &(inMap->second);
  return roc;
}


std::unordered_map<uint32_t, unsigned int>  SiPixelFedCablingMap::det2fedMap() const {
  std::unordered_map<uint32_t, unsigned int> result;
  for (auto im = theMap.begin(); im != theMap.end(); ++im) {
    result[im->second.rawId()] = im->first.fed;  // we know: a det is in only one fed!
  }
  return result;
}

std::map< uint32_t,std::vector<sipixelobjects::CablingPathToDetUnit> > SiPixelFedCablingMap::det2PathMap() const {
  std::map< uint32_t,std::vector<sipixelobjects::CablingPathToDetUnit> > result;
  for (auto im = theMap.begin(); im != theMap.end(); ++im) {
    CablingPathToDetUnit path = {im->first.fed, im->first.link, im->first.roc};
    result[im->second.rawId()].push_back(path);
  }
  return result;
}


std::vector<sipixelobjects::CablingPathToDetUnit> SiPixelFedCablingMap::pathToDetUnit(
      uint32_t rawDetId) const {

  std::vector<sipixelobjects::CablingPathToDetUnit> result;
  for (auto im = theMap.begin(); im != theMap.end(); ++im) {
    if(im->second.rawId()==rawDetId ) {
      CablingPathToDetUnit path = {im->first.fed, im->first.link, im->first.roc};
      result.push_back(path);
    }
  }
  return result;
}

