#include "CalibFormats/SiStripObjects/interface/SiStripReverseReadoutCabling.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include <iostream>
#include <vector>

using namespace std;


SiStripReverseReadoutCabling::SiStripReverseReadoutCabling(){}

SiStripReverseReadoutCabling::SiStripReverseReadoutCabling(const SiStripReadoutCabling * cabling){

  const vector<unsigned short> feds = cabling->getFEDs();

//these numbers should be put in an include file
  unsigned short MaxFedId=1023;
  unsigned short MaxFedCh=95;

  if(feds.size() > MaxFedId){
    cout<<"SiStripReadoutCabling::SiStripReadoutCabling - Error: trying to construct FED cabling structure with FED ids higher than maximum allowed value ("<<MaxFedId<<"). Throwing exception (string)"<<endl; 
    throw string("Exception thrown in SiStripReadoutCabling::SiStripReadoutCabling ");
  }

  for(vector<unsigned short>::const_iterator it = feds.begin(); it!= feds.end(); it++){
    

    const vector< SiStripReadoutCabling::APVPairId > apvpairs = cabling->getFEDAPVPairs(*it);

    unsigned short fed_ch=0;    

    for(vector< SiStripReadoutCabling::APVPairId >::const_iterator apvit = apvpairs.begin(); apvit!=apvpairs.end(); apvit++ ){


      if(fed_ch > MaxFedCh){
	cout<<"SiStripReadoutCabling::SiStripReadoutCabling - Error: trying to construct FED "<< *it <<" cabling with number of channels higher than maximum allowed value ("<<MaxFedCh<<"). Throwing exception (string)"<<endl; 
	throw string("Exception thrown in SiStripReadoutCabling::SiStripReadoutCabling ");
      }

      theDetConnections[(*apvit).first][(*apvit).second] = pair<unsigned short, unsigned short>(*it,fed_ch);
      fed_ch++;
     
    }

  }

}

SiStripReverseReadoutCabling::~SiStripReverseReadoutCabling(){}



const SiStripReadoutCabling::FEDChannelId & SiStripReverseReadoutCabling::getFEDChannel(const uint32_t det_id , unsigned short apvpair_id) const {

  static const SiStripReadoutCabling::FEDChannelId stdRetVal(99999, 99);

  const map<unsigned short, SiStripReadoutCabling::FEDChannelId> apvs = getAPVPairs(det_id);

  const map<unsigned short, SiStripReadoutCabling::FEDChannelId>::const_iterator it =  apvs.find(apvpair_id);

  if(it != apvs.end() ) return it->second;
  else{

    cout<<"SiStripReadoutCabling::getFEDChannel(..) - WARNING : requested Det id - Apv pair ("<< det_id<<"-"<< apvpair_id<<" ) is not present. Returning invalid FED channel"<<endl;
    return stdRetVal;

  }
 
}

const SiStripReadoutCabling::FEDChannelId & SiStripReverseReadoutCabling::getFEDChannel(const SiStripReadoutCabling::APVPairId & apvpair) const {

  return getFEDChannel(apvpair.first, apvpair.second);

}

const map<unsigned short, SiStripReadoutCabling::FEDChannelId> & SiStripReverseReadoutCabling::getAPVPairs(const uint32_t det_id) const {

  static const map<unsigned short, SiStripReadoutCabling::FEDChannelId> stdRetVal;

  map<uint32_t, map<unsigned short, SiStripReadoutCabling::FEDChannelId> >::const_iterator it = theDetConnections.find(det_id);

  if(it != theDetConnections.end() ) return it->second;
  else{

    cout<<"SiStripReadoutCabling::getAPVPairs(..) - WARNING : requested Det id ("<< det_id<<") is not present. Returning empty APV pair map"<<endl;
    return stdRetVal;

  }

}


const map<uint32_t, map<unsigned short, SiStripReadoutCabling::FEDChannelId> > & SiStripReverseReadoutCabling::getDetConnections() const {

  return theDetConnections;

}


void SiStripReverseReadoutCabling::debug() const{

  for (map<uint32_t, map<unsigned short, SiStripReadoutCabling::FEDChannelId> >::const_iterator i = theDetConnections.begin() ; i !=theDetConnections.end() ; i++) {

    for(map<unsigned short, SiStripReadoutCabling::FEDChannelId>::const_iterator j = (i->second).begin() ; j != (i->second).end() ; j++) {

      cout <<" Det Id # "<< i->first  <<" " 
	   <<" Apv pair # " << j->first <<" attached to FED "
	   << j->second.first <<" in channel "
	   << j->second.second << endl;
    }
  }
  
}

EVENTSETUP_DATA_REG(SiStripReverseReadoutCabling);
