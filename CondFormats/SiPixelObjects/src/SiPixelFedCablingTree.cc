#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include <sstream>
#include <iostream>

using namespace std;
using namespace sipixelobjects;

typedef std::map<int, SiPixelFedCablingTree::PixelFEDCabling>::const_iterator IMAP;

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
    unsigned int fedId, unsigned int linkId, unsigned int rocId) const
{
  const PixelROC* roc = 0;
  const PixelFEDCabling * aFed = fed(fedId);
  if (aFed) {
    const  PixelFEDLink * aLink = aFed->link(linkId);
    if (aLink) roc = aLink->roc(rocId);
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
