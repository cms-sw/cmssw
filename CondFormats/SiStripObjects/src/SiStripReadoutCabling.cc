#include "CondFormats/SiStripObjects/interface/SiStripReadoutCabling.h"
#include <iostream>
#include <string>


SiStripReadoutCabling::SiStripReadoutCabling(const vector< vector< APVPairId > >& cabling){

  //these numbers should be put in an include file
  unsigned short MaxFedId=1023;
  unsigned short MaxFedCh=95;

  if (cabling.size() >MaxFedId ){
    cout<<"SiStripReadoutCabling::SiStripReadoutCabling - Error: trying to construct FED cabling structure with FED ids higher than maximum allowed value ("<<MaxFedId<<"). Throwing exception (string)"<<endl; 
    throw string("Exception thrown");
  }
 

  unsigned short fed=0;

  for(vector< vector< APVPairId > >::const_iterator fedit = cabling.begin(); fedit!= cabling.end(); fedit++ ){

    if(fedit->size() > MaxFedCh){
      cout<<"SiStripReadoutCabling::SiStripReadoutCabling - Error: trying to construct FED cabling with number of channels higher than maximum allowed value ("<<MaxFedCh<<"). Throwing exception (string)"<<endl; 
      throw string("Exception thrown");
    }


    if(fedit->size()) theFEDs.push_back(fed);
    theFEDConnections.push_back(*fedit);
    fed++;

  }

}

SiStripReadoutCabling::SiStripReadoutCabling(){}

SiStripReadoutCabling::~SiStripReadoutCabling(){}


const SiStripReadoutCabling::APVPairId & SiStripReadoutCabling::getAPVPair(unsigned short fed_id, unsigned short fed_channel) const {

  static APVPairId stdRetVal(0, 99);
  //could be faster hardwiring  95 below instead of size() method. These numbers, including number of strips in an APV and others, should go into a common include file
  if ( fed_id < theFEDConnections.size() ) {
    if ( fed_channel < theFEDConnections[fed_id].size() ) {
      return theFEDConnections[fed_id][fed_channel];
    } else {
      cout << "SiStripReadoutCabling::getAPVPair(...) : " 
	   << "ERROR : \"fed_channel > detUnitMap_[fed_id].size()\": "
	   << fed_channel << " > " << theFEDConnections[fed_id].size();
      return stdRetVal;
    }
  } else {
    cout << "SiStripReadoutCabling::getAPVPair(...) : " 
	 << "ERROR : \"fed_id > theFEDConnections.size()\": " 
	 << fed_id << " > " << theFEDConnections.size();
    return stdRetVal;
  }
}


const SiStripReadoutCabling::APVPairId & SiStripReadoutCabling::getAPVPair(const FEDChannelId & fedch_id) const{

  return getAPVPair(fedch_id.first, fedch_id.second);

}

const vector<unsigned short> & SiStripReadoutCabling::getFEDs() const{

  return theFEDs;

}

const vector< vector< SiStripReadoutCabling::APVPairId >  > & SiStripReadoutCabling::getFEDConnections() const{

  return theFEDConnections;

}
