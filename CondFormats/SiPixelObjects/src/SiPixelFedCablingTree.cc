#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include <sstream>
#include <iostream>

using namespace std;
using namespace sipixelobjects;

typedef std::map<int, SiPixelFedCablingTree::PixelFEDCabling>::const_iterator IMAP;

std::vector<sipixelobjects::CablingPathToDetUnit> SiPixelFedCablingTree::pathToDetUnit(
      uint32_t rawDetId) const
{
  std::vector<sipixelobjects::CablingPathToDetUnit> result;
  typedef std::map<int, PixelFEDCabling>::const_iterator IM;
  for (IM im = theFedCablings.begin(); im != theFedCablings.end(); ++im) {
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

void SiPixelFedCablingTree::addFed(const PixelFEDCabling & f)
{
  int id = f.id();
  theFedCablings[id] = f;
}

const PixelFEDCabling * SiPixelFedCablingTree::fed(unsigned int id) const
{
  IMAP  it = theFedCablings.find(id);
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

int SiPixelFedCablingTree::checkNumbering() const
{
  int status = 0;
  for (std::map<int, PixelFEDCabling>::const_iterator im = theFedCablings.begin();
       im != theFedCablings.end(); ++im) {
    if (im->first != static_cast<int>( im->second.id())) {
      status = 1;
      std::cout <<  "PROBLEM WITH FED ID!!" << im->first <<" vs: "<< im->second.id() << std::endl; 
    }
    im->second.checkLinkNumbering();
  }
  return status;
}
