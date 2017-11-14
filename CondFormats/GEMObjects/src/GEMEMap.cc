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
  uint16_t amcId = 1; //amc
  uint16_t gebId = 1; 
  romap->addAMC(amcId);
	
  for (int re = -1; re <= 1; re = re+2) {
    for (int st = 1; st<=2; ++st) {
      // for (int ch = GEMDetId::minChamberId; ch<=GEMDetId::maxChamberId; ++ch) {
      // 	for (int ly = 1; ly<=GEMDetId::maxLayerId; ++ly) {
      for (int ch = 1; ch<=36; ++ch) {
	for (int ly = 1; ly<=2; ++ly) {
	  
	  // 1 geb per chamber
	  // 24 gebs per amc
	  if (gebId > 25){
	    gebId = 1;
	    amcId++;
	    romap->addAMC(amcId);
	  }

	  romap->addAMC2GEB(amcId, gebId);
	  
	  GEMDetId chamDetId(re, 1, st, ly, ch, 0);
	  uint32_t chamberId = (amcId << 5) | gebId;	  
	  romap->add(chamDetId,chamberId);
	  romap->add(chamberId,chamDetId);
	  
	  uint16_t chipId = 0;	 	  
	  for (int roll = 1; roll<=GEMDetId::maxRollId; ++roll) {
	    int maxVFat = 3;
	    if (st == 2){
	      maxVFat = 6;
	    }
	    
	    int stripId = 0;
	    GEMDetId gemId(re, 1, st, ly, ch, roll);
	    
	    for (int nVfat = 0; nVfat < maxVFat; ++nVfat){
	      chipId++;
	      
	      for (unsigned chan = 0; chan < 128; ++chan){	    
		GEMROmap::dCoord dc;
		dc.stripId = ++stripId;
		dc.gemDetId = gemId;

		uint32_t vfatId = (amcId << 17) | (gebId << 12) | chipId;
		
		GEMROmap::eCoord ec;
		ec.vfatId =  vfatId;
		ec.channelId = chan;
		romap->add(ec,dc);
		romap->add(dc,ec);

	      }
	    }	    
	  }

	  gebId++;

	}
      }
    }
  }
  
  return romap;
}
