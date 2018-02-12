#include "CondFormats/GEMObjects/interface/ME0EMap.h"
#include "CondFormats/GEMObjects/interface/ME0ROmap.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

ME0EMap::ME0EMap():
  theVersion("") {}

ME0EMap::ME0EMap(const std::string & version):
  theVersion(version) {}

ME0EMap::~ME0EMap() {}

const std::string & ME0EMap::version() const{
  return theVersion;
}

void ME0EMap::convert(ME0ROmap & romap) {

  for (auto imap : theVFatMaptype_){
    for (unsigned int ix=0;ix<imap.vfatId.size();ix++){
      ME0ROmap::eCoord ec;
      ec.vfatId = imap.vfatId[ix];
      ec.gebId = imap.gebId[ix];
      ec.amcId = imap.amcId[ix];      

      ME0ROmap::dCoord dc;
      dc.me0DetId = ME0DetId(imap.z_direction[ix], imap.depth[ix], imap.sec[ix], imap.iEta[ix]);
      dc.vfatType = imap.vfatType[ix]; 
      dc.iPhi = imap.iPhi[ix];

      romap.add(ec,dc);
      romap.add(dc,ec);
    }
  }
  
  for (auto imap : theVfatChStripMap_){
    for (unsigned int ix=0;ix<imap.vfatType.size();ix++){
      ME0ROmap::channelNum cMap;
      cMap.vfatType = imap.vfatType[ix];
      cMap.chNum = imap.vfatCh[ix];

      ME0ROmap::stripNum sMap;
      sMap.vfatType = imap.vfatType[ix];
      sMap.stNum = imap.vfatStrip[ix];

      romap.add(cMap, sMap);
      romap.add(sMap, cMap);
    }
  }
}

void ME0EMap::convertDummy(ME0ROmap & romap) {
  // 12 bits for vfat, 5 bits for geb, 8 bit long GLIB serial number
  uint16_t amcId = 1; //amc
  uint16_t gebId = 0; 
	
  for (int re = -1; re <= 1; re = re+2) {
      
    for (int ch = 1; ch<=ME0DetId::maxChamberId; ++ch) {
      for (int ly = 1; ly<=ME0DetId::maxLayerId; ++ly) {
	// 1 geb per chamber
	gebId++;	  	  	  
	uint16_t chipId = 0;	 	  
	for (int roll = 1; roll<=ME0DetId::maxRollId; ++roll) {
	    
	  ME0DetId me0Id(re, ly, ch, roll);

	  for (int nphi = 1; nphi <= maxVFat_; ++nphi){
	    chipId++;
	      
	    ME0ROmap::eCoord ec;
	    ec.vfatId = chipId;
	    ec.gebId = gebId;
	    ec.amcId = amcId;

	    ME0ROmap::dCoord dc;
	    dc.me0DetId = me0Id;
	    dc.vfatType = 1;
	    dc.iPhi = nphi;

	    romap.add(ec,dc);
	    romap.add(dc,ec);
	  }
	}
	// 5 bits for geb
	if (gebId == maxGEBs_){
	  // 24 gebs per amc
	  gebId = 0;
	  amcId++;
	}
      }
    }
  }

  for (int i = 0; i < maxChan_; ++i){
    // only 1 vfat type for dummy map
    ME0ROmap::channelNum cMap;
    cMap.vfatType = 1;
    cMap.chNum = i;

    ME0ROmap::stripNum sMap;
    sMap.vfatType = 1;
    sMap.stNum = i+1;

    romap.add(cMap, sMap);
    romap.add(sMap, cMap);
  }
}
