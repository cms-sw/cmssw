
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
      ec.vfatId= imap->vfatId[ix] - 0xf000;// chip ID is 12 bits
      ec.channelId=imap->vfat_chnnel_number[ix];
      GEMROmap::dCoord dc;
      dc.stripId = 1 + imap->strip_number[ix]+(imap->iPhi[ix]-1)%3*128;
      dc.gemDetId = GEMDetId(imap->z_direction[ix], 1, std::abs(imap->z_direction[ix]), imap->depth[ix], imap->sec[ix], imap->iEta[ix]); 
      romap->add(ec,dc);
      romap->add(dc,ec);
    }
  }
  return romap;
}

GEMROmap* GEMEMap::convertDummy() const{
  GEMROmap* romap=new GEMROmap();

  uint16_t gebId = 0;
  for (int re = -1; re <= 1; re = re+2) {
    for (int st = 1; st<=2; ++st) {
      for (int ch = 1; ch<=36; ++ch) {
	gebId++;
	GEMDetId gebDetId(re, 1, st, 0, ch, 0);
	romap->add(gebId,gebDetId);
	romap->add(gebDetId,gebId);
	
	uint16_t vfatId = 0;
	
	for (int ly = 1; ly<=2; ++ly) {
	  for (int roll = 1; roll<=8; ++roll) {
	    GEMDetId gemId(re, 1, st, ly, ch, roll);
	    int maxVFat = 3;
	    if (st == 2){
	      if (ch > 18) continue;
	      maxVFat = 6;
	    }
	    int stripId = 0;
	    for (int nVfat = 0; nVfat < maxVFat; ++nVfat){
	      vfatId++;
	      for (unsigned chan = 0; chan < 128; ++chan){
		GEMROmap::eCoord ec;
		ec.vfatId = vfatId | gebId << 12;
		ec.channelId = chan;
	    
		GEMROmap::dCoord dc;
		dc.stripId = ++stripId;
		dc.gemDetId = gemId;
		romap->add(ec,dc);
		romap->add(dc,ec);
	      }
	    }
	    
	  }
	}
      }
    }
  }
  
  return romap;
}
