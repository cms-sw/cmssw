#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"

#include <vector>
#include <iostream>
#include <algorithm>

using namespace sipixelobjects;

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
// std::cout << "HERE --- SiPixelFedCablingMap CTOR" << std::endl;
  
  std::vector<const PixelFEDCabling *> fedList = cab->fedList();
  for (std::vector<const PixelFEDCabling *>::const_iterator ifed=fedList.begin();
   ifed != fedList.end(); ifed++) {
    int fed = (**ifed).id();
    int numLink = (**ifed).numberOfLinks();
    for (int link=1; link <= numLink; link++) {
      const PixelFEDLink * pLink = (**ifed).link(link); 
      if (pLink==0) continue;
      int linkId = static_cast<int>(pLink->id());
      if (linkId != 0 && linkId!= link) 
          std::cout << "PROBLEM WITH LINK NUMBER!!!!" << std::endl;
      int numberROC = pLink->numberOfROCs(); 
      for (int roc=1; roc <= numberROC; roc++) {
        const PixelROC * pROC = pLink->roc(roc);
        if (pROC==0) continue;
        if (static_cast<int>(pROC->idInLink()) != roc) 
            std::cout << "PROBLEM WITH ROC NUMBER!!!!" << std::endl;
        Key key = {static_cast<unsigned int>(fed),
                   static_cast<unsigned int>(link),
                   static_cast<unsigned int>(roc)}; 
        theMap[key] = (*pROC);
      }
    } 
  }  
}

SiPixelFedCablingTree * SiPixelFedCablingMap::cablingTree() const
{
  SiPixelFedCablingTree * tree = new SiPixelFedCablingTree(theVersion); 
  for (Map::const_iterator im = theMap.begin(); im != theMap.end(); im++) {
    const sipixelobjects::PixelROC & roc = im->second;
    unsigned int fedId = im->first.fed;
    unsigned int linkId = im->first.link;
    tree->addItem(fedId, linkId,  roc);
  }
  return tree;
}

std::vector<unsigned int> SiPixelFedCablingMap::fedIds() const
{
  std::vector<unsigned int> result;
  for (Map::const_iterator im = theMap.begin(); im != theMap.end(); im++) {
    unsigned int fedId = im->first.fed;
    if (find(result.begin(),result.end(),fedId) == result.end()) result.push_back(fedId);
  }
  return result;
}

const sipixelobjects::PixelROC* SiPixelFedCablingMap::findItem(
    const sipixelobjects::CablingPathToDetUnit & path) const
{
  const PixelROC* roc = 0;
  Key key = {path.fed, path.link, path.roc};
  Map::const_iterator inMap = theMap.find(key);
  if (inMap!= theMap.end()) roc = &(inMap->second);
  return roc;
}

std::vector<sipixelobjects::CablingPathToDetUnit> SiPixelFedCablingMap::pathToDetUnit(
      uint32_t rawDetId) const
{
  std::vector<sipixelobjects::CablingPathToDetUnit> result;
  for (Map::const_iterator im = theMap.begin(); im != theMap.end(); ++im) {
    if(im->second.rawId()==rawDetId ) {
      CablingPathToDetUnit path = {im->first.fed, im->first.link, im->first.roc};
      result.push_back(path);
    }
  }
  return result;
}

