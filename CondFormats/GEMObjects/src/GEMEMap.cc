#include <iostream>
#include "CondFormats/GEMObjects/interface/GEMEMap.h"
#include "CondFormats/GEMObjects/interface/GEMROmap.h"

GEMEMap::GEMEMap():
  theVersion("") {}

GEMEMap::GEMEMap(const std::string & version):
  theVersion(version) {}

GEMEMap::~GEMEMap() {}

const std::string & GEMEMap::version() const{
  return theVersion;
}

GEMROmap* GEMEMap::convert() const{
  // Be careful, this currently does not work since EMap and VFatMapInPos are not filled 
  GEMROmap* romap=new GEMROmap();

  for (std::vector<GEMEMap::GEMEMapItem>::const_iterator i=this->theEMapItem.begin(); i<this->theEMapItem.end(); i++){

    std::vector<int>::const_iterator p=i->positions.begin();
    for (std::vector<int>::const_iterator v=i->VFatIDs.begin(); v<i->VFatIDs.end(); v++,p++){
      int position = *p;
      int sPhi = position/8;

      std::vector<GEMEMap::GEMVFatMapInPos>::const_iterator ipos;
      int mid=-1;
      for (ipos=this->theVFatMapInPos.begin();ipos<this->theVFatMapInPos.end();ipos++){
	if (ipos->position == position){
	  mid = ipos->VFATmapTypeId;

	  std::vector<GEMEMap::GEMVFatMaptype>::const_iterator imap;
	  for (imap=this->theVFatMaptype.begin(); imap<this->theVFatMaptype.end();imap++){
	    if (mid == imap->VFATmapTypeId){
	      for (unsigned int ix=0;ix<imap->strip_number.size();ix++){
		GEMROmap::eCoord ec;
		ec.chamberId=i->ChamberID;
		ec.vfatId=*v;
		ec.channelId=imap->vfat_chnnel_number[ix];
		GEMROmap::dCoord dc;
		dc.etaId = 8-position%8;
		dc.stripId = imap->strip_number[ix]+sPhi*128;
		romap->add(ec,dc);
	      }
	    }
	  }
	}
      }
    }
  }
  return romap;
}

GEMROmap* GEMEMap::convertCS() const { //uint64_t ChipID[24]) const{
  GEMROmap* romap=new GEMROmap();

  uint64_t ChipID[24]={ // Hardcoded, numbers from Jared, this should change
    0xfa20,         0xff48,        0xffcb,        0xffec,        0xfa40,        0xffdc,        0xf747,        0xff40,
    0xfffc,        0xffe7,        0xdead,        0xff83,        0xff87,        0xf6bc,        0xf75b,        0xff4c,
    0xf99b,        0xff24,        0xff9b,        0xff2c,        0xfa28,        0xfa17,        0xff6b,        0xfa2c
  };


  for (int position=0; position<24; position++){
    int sPhi = position/8;
    std::vector<GEMEMap::GEMVFatMaptype>::const_iterator imap;
    for (imap=this->theVFatMaptype.begin(); imap<this->theVFatMaptype.end();imap++){
      for (unsigned int ix=0;ix<imap->strip_number.size();ix++){
	if(position!=imap->vfat_position[ix]) continue;
	GEMROmap::eCoord ec;
	ec.chamberId=31;//i->ChamberID;  // This is dummy for now
	ec.vfatId=ChipID[position];
	ec.channelId=imap->vfat_chnnel_number[ix];
	GEMROmap::dCoord dc;
	dc.etaId = 8-position%8;
	dc.stripId = imap->strip_number[ix]+sPhi*128;
	romap->add(ec,dc);

	/*if(ec.channelId==58) {
	  std::cout <<"Chamber "<<ec.chamberId<<" vfat 0x"<<std::hex<<ec.vfatId<<std::dec<<" chan="<<ec.channelId
	  <<" correspond to eta="<<dc.etaId<<" strip="<<dc.stripId<<std::endl;

	  std::cout <<
	  " subdet "<<imap->subdet[ix] <<
	  //" sector "<<imap->sector[ix] <<
	  " tscol "<<imap->tscol[ix] <<
	  " tsrow "<<imap->tsrow[ix] <<
	  " type "<<imap->type[ix] <<
	  " z_direction "<<imap->z_direction[ix] <<
	  " iEta "<<imap->iEta[ix] <<
	  " iPhi "<<imap->iPhi[ix] <<
	  " depth "<<imap->depth[ix] <<
	  " vfat_position "<<imap->vfat_position[ix] <<
	  " strip_number "<<imap->strip_number[ix] <<
	  " vfat_chnnel_number "<<imap->vfat_chnnel_number[ix]<<
	  " px_connector_pin "<<imap->px_connector_pin[ix]<<std::endl;
	  }
	*/



      }
    }
  }

  return romap;
}


GEMROmap* GEMEMap::convertCSConfigurable(std::vector<unsigned long long>* vfats,std::vector<int>* slot) const{
  GEMROmap* romap=new GEMROmap();

  if(vfats->size()!=slot->size()){ std::cout<<"Wrong input configuration"<<std::endl; return romap;} 

  for (unsigned int i=0; i<slot->size(); i++){
    int position=slot->at(i);
    if(position==-1) continue;
    //                std::cout<<position<<"    "<<vfats->at(i)<<std::endl;

    std::vector<GEMEMap::GEMVFatMaptype>::const_iterator imap;

    for (imap=this->theVFatMaptype.begin(); imap<this->theVFatMaptype.end();imap++){
      for (unsigned int ix=0;ix<imap->strip_number.size();ix++){
	if(position!=imap->vfat_position[ix]) continue;
	GEMROmap::eCoord ec;
	ec.chamberId=31;//i->ChamberID;
	ec.vfatId=(uint64_t)vfats->at(i);  // compiler issues with unsigned long long and uint64_t depending on architecture
	ec.channelId=imap->vfat_chnnel_number[ix];
	GEMROmap::dCoord dc;
	dc.etaId =imap->iEta[ix];
	int sPhi=imap->iPhi[ix];
	dc.stripId = imap->strip_number[ix]+(sPhi-1)*128;
	romap->add(ec,dc);

	//                                                            if(ec.vfatId==0xf9e8)    std::cout <<"Chamber "<<ec.chamberId<<" vfat 0x"<<std::hex<<ec.vfatId<<std::dec<<" chan="<<ec.channelId
	//                                                                        <<" correspond to eta="<<dc.etaId<<" strip="<<dc.stripId<<std::endl;

      }
    }
  }

  return romap;

}






