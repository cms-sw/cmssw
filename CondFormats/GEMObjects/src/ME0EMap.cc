
#include <iostream>
#include "CondFormats/GEMObjects/interface/ME0EMap.h"
#include "CondFormats/GEMObjects/interface/ME0ROmap.h"

#include <DataFormats/MuonDetId/interface/ME0DetId.h>

ME0EMap::ME0EMap():
  theVersion("") {}

ME0EMap::ME0EMap(const std::string & version):
  theVersion(version) {}

ME0EMap::~ME0EMap() {}

const std::string & ME0EMap::version() const{
  return theVersion;
}

ME0ROmap* ME0EMap::convert() const{
  ME0ROmap* romap=new ME0ROmap();

  std::vector<ME0EMap::ME0VFatMaptype>::const_iterator imap;
  for (imap=this->theVFatMaptype.begin(); imap<this->theVFatMaptype.end();imap++){
    for (unsigned int ix=0;ix<imap->strip_number.size();ix++){
      ME0ROmap::eCoord ec;
      ec.vfatId= imap->vfatId[ix];
      ec.channelId=imap->vfat_chnnel_number[ix];
      ME0ROmap::dCoord dc;
      dc.stripId = 1 + imap->strip_number[ix]+(imap->vfat_position[ix]-1)%3*128;
      dc.gemDetId = ME0DetId(imap->z_direction[ix],
			     imap->depth[ix],
			     imap->sec[ix],
			     imap->iEta[ix]); 
      romap->add(ec,dc);
      romap->add(dc,ec);
    }
  }
  return romap;
}
