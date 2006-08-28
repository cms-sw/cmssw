#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include <sstream>

using namespace std;

void SiPixelFedCablingMap::addFed(const PixelFEDCabling & f)
{
  theFedCablings.push_back(f);
}

const PixelFEDCabling * SiPixelFedCablingMap::fed(unsigned int id) const
{
  return (id >= 0 && id < theFedCablings.size() )? &theFedCablings[id] : 0; 
}

string SiPixelFedCablingMap::print(int depth) const
{
  ostringstream out;
  typedef vector<PixelFEDCabling>::const_iterator IT;
  if ( depth-- >=0 ) {
    out << "== SiPixelFedCablingMap, version: "<< theVersion << endl;
    for(IT it=theFedCablings.begin(); it != theFedCablings.end(); it++) {
      out << (*it).print(depth);
    }
  }
  out << endl;
  return out.str();
}
