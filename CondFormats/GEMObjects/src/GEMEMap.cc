#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "CondFormats/GEMObjects/interface/GEMROmap.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

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

  for (auto imap : theVFatMaptype){
    for (unsigned int ix=0;ix<imap.strip_number.size();ix++){
      GEMROmap::eCoord ec;
      ec.vfatId= imap.vfatId[ix] & chipIdMask_;// chip ID is 12 bits
      ec.channelId=imap.vfat_chnnel_number[ix];

      int st = std::abs(imap.z_direction[ix]);
      int maxVFat = maxVFatGE11_;
      if (st == 2) maxVFat = maxVFatGE21_;      
      
      GEMROmap::dCoord dc;
      dc.stripId = 1 + imap.strip_number[ix]+(imap.iPhi[ix]-1)%maxVFat*maxChan_;
      dc.gemDetId = GEMDetId(imap.z_direction[ix], 1, st, imap.depth[ix], imap.sec[ix], imap.iEta[ix]);
      
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
    for (int st = GEMDetId::minStationId; st<=GEMDetId::maxStationId; ++st) {
      int maxVFat = maxVFatGE11_;
      if (st == 2) maxVFat = maxVFatGE21_;      
      
      for (int ch = 1; ch<=GEMDetId::maxChamberId; ++ch) {
      	for (int ly = 1; ly<=GEMDetId::maxLayerId; ++ly) {
	  
	  // 1 geb per chamber
	  // 24 gebs per amc
	  // make new amcId once 24 gebs are used up
	  if (gebId > maxGEBs_){
	    gebId = 1;
	    amcId++;
	    romap->addAMC(amcId);
	  }

	  romap->addAMC2GEB(amcId, gebId);
	  
	  GEMDetId chamDetId(re, 1, st, ly, ch, 0);
	  uint32_t chamberId = (amcId << gebIdBits_) | gebId;	  
	  romap->add(chamDetId,chamberId);
	  romap->add(chamberId,chamDetId);
	  
	  uint16_t chipId = 0;	 	  
	  for (int roll = 1; roll<=GEMDetId::maxRollId; ++roll) {
	    
	    int stripId = 0;
	    GEMDetId gemId(re, 1, st, ly, ch, roll);
	    
	    for (int nVfat = 0; nVfat < maxVFat; ++nVfat){
	      chipId++;
	      
	      for (unsigned chan = 0; chan < maxChan_; ++chan){
		GEMROmap::dCoord dc;
		dc.stripId = ++stripId;
		dc.gemDetId = gemId;

		// make 1 full vfat ID from amc + geb + chip Ids
		uint32_t vfatId = (amcId << (gebIdBits_+chipIdBits_)) | (gebId << chipIdBits_) | chipId;
		
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
