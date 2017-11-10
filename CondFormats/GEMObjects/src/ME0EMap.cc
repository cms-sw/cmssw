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
      ec.vfatId= imap->vfatId[ix] - 0xf000;// chip ID is 12 bits
      ec.channelId=imap->vfat_chnnel_number[ix];
      ME0ROmap::dCoord dc;
      dc.stripId = 1 + imap->strip_number[ix]+(imap->iPhi[ix]-1)%3*128;
      dc.me0DetId = ME0DetId(imap->z_direction[ix], imap->depth[ix], imap->sec[ix], imap->iEta[ix]); 
      romap->add(ec,dc);
      romap->add(dc,ec);
    }
  }
  return romap;
}

ME0ROmap* ME0EMap::convertDummy() const{
  ME0ROmap* romap=new ME0ROmap();
  uint16_t amcId = 1; //amc
  uint16_t gebId = 1; 
  romap->addAMC(amcId);
	
  for (int re = -1; re <= 1; re = re+2) {
    for (int ch = ME0DetId::minChamberId; ch<=ME0DetId::maxChamberId; ++ch) {
      for (int ly = 1; ly<=ME0DetId::maxLayerId; ++ly) {
	  
	// 1 geb per chamber
	// 24 gebs per amc
	if (gebId > 25){
	  gebId = 1;
	  amcId++;
	  romap->addAMC(amcId);
	}

	romap->addAMC2GEB(amcId, gebId);
	  
	ME0DetId chamDetId(re, ly, ch, 0);
	uint32_t chamberId = (amcId << 5) | gebId;	  
	romap->add(chamDetId,chamberId);
	romap->add(chamberId,chamDetId);
	  
	uint16_t chipId = 0;	 	  
	for (int roll = 1; roll<=ME0DetId::maxRollId; ++roll) {
	  int maxVFat = 12;// set to 12, not yet known
	    
	  int stripId = 0;
	  ME0DetId me0Id(re, ly, ch, roll);
	    
	  for (int nVfat = 0; nVfat < maxVFat; ++nVfat){
	    chipId++;
	      
	    for (unsigned chan = 0; chan < 128; ++chan){	    
	      ME0ROmap::dCoord dc;
	      dc.stripId = ++stripId;
	      dc.me0DetId = me0Id;

	      uint32_t vfatId = (amcId << 17) | (gebId << 12) | chipId;
		
	      ME0ROmap::eCoord ec;
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
  
  return romap;
}
