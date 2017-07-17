#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include <algorithm>
#include <sstream>
#include <iostream>

using namespace std;
using namespace sipixelobjects;

typedef std::unordered_map<int, SiPixelFedCablingTree::PixelFEDCabling>::const_iterator IMAP;

std::vector<sipixelobjects::CablingPathToDetUnit> SiPixelFedCablingTree::pathToDetUnit(
      uint32_t rawDetId) const
{
  std::vector<sipixelobjects::CablingPathToDetUnit> result;
  for (auto im = theFedCablings.begin(); im != theFedCablings.end(); ++im) {
    const PixelFEDCabling & aFed = im->second;
    for (unsigned int idxLink = 1; idxLink <= aFed.numberOfLinks(); idxLink++) {
      const PixelFEDLink * link = aFed.link(idxLink);
      if (!link) continue;
      unsigned int numberOfRocs = link->numberOfROCs();
      for(unsigned int idxRoc = 1; idxRoc <= numberOfRocs; idxRoc++) {
        const PixelROC * roc = link->roc(idxRoc);
        if (rawDetId == roc->rawId() ) {
          CablingPathToDetUnit path = {aFed.id(), link->id(), roc->idInLink()};
          result.push_back(path);
        } 
      }
    }
  }
  return result;
}

std::unordered_map<uint32_t, unsigned int> SiPixelFedCablingTree::det2fedMap() const {
  std::unordered_map<uint32_t, unsigned int> result;
  for (auto im = theFedCablings.begin(); im != theFedCablings.end(); ++im) {
    auto const & aFed = im->second;
    for (unsigned int idxLink = 1; idxLink <= aFed.numberOfLinks(); idxLink++) {
      auto link = aFed.link(idxLink);
      if (!link) continue;
      unsigned int numberOfRocs = link->numberOfROCs();
      for(unsigned int idxRoc = 1; idxRoc <= numberOfRocs; idxRoc++) {
        auto roc = link->roc(idxRoc);
        result[roc->rawId()]=aFed.id();  // we know that a det is in one fed only...
      }
    }
  }
  return result;
}

std::map< uint32_t,std::vector<sipixelobjects::CablingPathToDetUnit> > SiPixelFedCablingTree::det2PathMap() const {
  std::map< uint32_t,std::vector<sipixelobjects::CablingPathToDetUnit> > result;
  for (auto im = theFedCablings.begin(); im != theFedCablings.end(); ++im) {
    auto const & aFed = im->second;
    for (unsigned int idxLink = 1; idxLink <= aFed.numberOfLinks(); idxLink++) {
      auto link = aFed.link(idxLink);
      if (!link) continue;
      unsigned int numberOfRocs = link->numberOfROCs();
      for(unsigned int idxRoc = 1; idxRoc <= numberOfRocs; idxRoc++) {
        auto roc = link->roc(idxRoc);
        CablingPathToDetUnit path = {aFed.id(), link->id(), roc->idInLink()};
        result[roc->rawId()].push_back(path);
      }
    }
  }
  return result;
}

void SiPixelFedCablingTree::addFed(const PixelFEDCabling & f)
{
  int id = f.id();
  theFedCablings[id] = f;
}

const PixelFEDCabling * SiPixelFedCablingTree::fed(unsigned int id) const
{
  auto  it = theFedCablings.find(id);
  return ( it == theFedCablings.end() ) ? 0 : & (*it).second;
}

string SiPixelFedCablingTree::print(int depth) const
{
  ostringstream out;
  if ( depth-- >=0 ) {
    out << theVersion << endl;
    for(IMAP it=theFedCablings.begin(); it != theFedCablings.end(); it++) {
      out << (*it).second.print(depth);
    }
  }
  out << endl;
  return out.str();
}

std::vector<const PixelFEDCabling *> SiPixelFedCablingTree::fedList() const
{
  std::vector<const PixelFEDCabling *> result;
  for (IMAP im = theFedCablings.begin(); im != theFedCablings.end(); im++) {
    result.push_back( &(im->second) );
  }
  std::sort(result.begin(),result.end(),[](const PixelFEDCabling * a,const PixelFEDCabling * b){return a->id()<b->id();});
  return result;

}

void SiPixelFedCablingTree::addItem(unsigned int fedId, unsigned int linkId, const PixelROC& roc)
{
  PixelFEDCabling & cabling = theFedCablings[fedId];
  if (cabling.id() != fedId) cabling=PixelFEDCabling(fedId);
  cabling.addItem(linkId,roc);
}

const sipixelobjects::PixelROC* SiPixelFedCablingTree::findItem(
								const CablingPathToDetUnit & path) const
{
  const PixelROC* roc = 0;
  const PixelFEDCabling * aFed = fed(path.fed);
  if (aFed) {
    const  PixelFEDLink * aLink = aFed->link(path.link);
    if (aLink) roc = aLink->roc(path.roc);
  }
  return roc;
}


const sipixelobjects::PixelROC* SiPixelFedCablingTree::findItemInFed(
								const CablingPathToDetUnit & path, 
								const PixelFEDCabling * aFed) const
{
  const PixelROC* roc = 0;
  const  PixelFEDLink * aLink = aFed->link(path.link);
  if (aLink) roc = aLink->roc(path.roc);
  return roc;
}


int SiPixelFedCablingTree::checkNumbering() const
{
  int status = 0;
  for (auto im = theFedCablings.begin(); im != theFedCablings.end(); ++im) {
    if (im->first != static_cast<int>( im->second.id())) {
      status = 1;
      std::cout <<  "PROBLEM WITH FED ID!!" << im->first <<" vs: "<< im->second.id() << std::endl; 
    }
    im->second.checkLinkNumbering();
  }
  return status;
}
