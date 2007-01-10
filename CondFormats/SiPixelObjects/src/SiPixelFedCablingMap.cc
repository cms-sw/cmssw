#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include <sstream>
#include <iostream>

using namespace std;
using namespace sipixelobjects;

typedef std::map<int, SiPixelFedCablingMap::PixelFEDCabling>::const_iterator IMAP;

void SiPixelFedCablingMap::addFed(const PixelFEDCabling & f)
{
  int id = f.id();
  theFedCablings[id] = f;
}

const PixelFEDCabling * SiPixelFedCablingMap::fed(unsigned int id) const
{
  IMAP  it = theFedCablings.find(id);
  return ( it == theFedCablings.end() ) ? 0 : & (*it).second;
}

string SiPixelFedCablingMap::print(int depth) const
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

std::vector<const PixelFEDCabling *> SiPixelFedCablingMap::fedList() const
{
  std::vector<const PixelFEDCabling *> result;
  for (IMAP im = theFedCablings.begin(); im != theFedCablings.end(); im++) {
    result.push_back( &(im->second) );
  }
  return result;

}

void SiPixelFedCablingMap:: addItem(int fedId, int linkId, const PixelROC& roc)
{
  PixelFEDCabling & cabling = theFedCablings[fedId];
  if (cabling.id() != fedId) cabling=PixelFEDCabling(fedId);
  cabling.addItem(linkId,roc);
}
