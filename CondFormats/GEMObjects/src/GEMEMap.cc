
#include <iostream>
#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "CondFormats/GEMObjects/interface/GEMROmap.h"

#include <DataFormats/MuonDetId/interface/GEMDetId.h>

GEMEMap::GEMEMap():
  theVersion("") {}

GEMEMap::GEMEMap(const std::string & version):
  theVersion(version) {}

GEMEMap::~GEMEMap() {}

const std::string & GEMEMap::version() const{
  return theVersion;
}

GEMROmap* GEMEMap::convert() const{
  GEMROmap* romap=new GEMROmap();

  std::vector<GEMEMap::GEMVFatMaptype>::const_iterator imap;
  for (imap=this->theVFatMaptype.begin(); imap<this->theVFatMaptype.end();imap++){
    for (unsigned int ix=0;ix<imap->strip_number.size();ix++){
      GEMROmap::eCoord ec;
      ec.vfatId= imap->vfatId[ix];
      ec.channelId=imap->vfat_chnnel_number[ix];
      GEMROmap::dCoord dc;
      dc.stripId = 1 + imap->strip_number[ix]+(imap->vfat_position[ix]-1)%3*128;
      dc.gemDetId = GEMDetId(imap->z_direction[ix], 1, std::abs(imap->z_direction[ix]), imap->depth[ix], imap->sec[ix], imap->iEta[ix]); 
      romap->add(ec,dc);
    }
  }
  return romap;
}





