// -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripDetCabling
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed Mar 22 12:24:33 CET 2006
// $Id$
//

// system include files

// user include files
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

using namespace std;

SiStripDetCabling::SiStripDetCabling()
{
}

// SiStripDetCabling::SiStripDetCabling(const SiStripDetCabling& rhs)
// {
//    // do actual copying here;
// }

SiStripDetCabling::~SiStripDetCabling()
{
}


SiStripDetCabling::SiStripDetCabling(const SiStripFedCabling& fedcabling) : detCabling_()
{ // create detCabling_, loop over vector of FedChannelConnection, either make new element of map, or add to appropriate vector of existing map element
  // get feds list (vector) from fedcabling object - these are the active FEDs
  const vector<uint16_t>& feds = fedcabling.feds();
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) { // iterate over active feds, get all their FedChannelConnection-s
    const vector<FedChannelConnection>& conns = fedcabling.connections( *ifed );
    vector<FedChannelConnection>::const_iterator iconn;
    for ( iconn = conns.begin(); iconn != conns.end(); iconn++ ) { // loop over FedChannelConnection objects
      addDevices(*iconn); // leave separate method, in case you will need to add devices also after constructing
    }
  }
}

void SiStripDetCabling::addDevices(const FedChannelConnection & conn){ // could also return an integer value/flag to tell whether device was already present
// separate method in case need to add devices/cabling after object already constructed
  const uint32_t& fedchannelconnection_detid = conn.detId(); // detId of FedChannelConnection that is looped over
  // is this detid already in the map?
  map< uint32_t, vector<FedChannelConnection> >::iterator detcabl_it = detCabling_.find(fedchannelconnection_detid);
  if( ! (detcabl_it==detCabling_.end()) ){      // is already in map -- DKwarn : add constistency checks here for example agreement with already existing dcuId?
    ( detcabl_it->second ).push_back(conn); // add one FedChannelConnection to vector mapped to this detid
  }else{                                      // is not in map, create vector with 1 element and put into map
    vector<FedChannelConnection> local_fedchannelconnection; local_fedchannelconnection.push_back(conn);
    detCabling_[fedchannelconnection_detid] = local_fedchannelconnection;
  }
}

const std::vector<uint32_t> & SiStripDetCabling::getActiveDetectorsRawIds() const{
 static vector<uint32_t> local_active;
 for(map< uint32_t, vector<FedChannelConnection> >::const_iterator detcabl_it = detCabling_.begin(); detcabl_it!=detCabling_.end(); detcabl_it++){
   local_active.push_back(detcabl_it->first);
 }
 return local_active; // is empty if no elements in detCabling_
}


const vector<FedChannelConnection>& SiStripDetCabling::getConnections(uint32_t det_id ) const{ // return all connections corresponding to one det_id
  map< uint32_t, vector<FedChannelConnection> >::const_iterator detcabl_it = detCabling_.find(det_id); // has to be const_iterator because this function cannot change data members
  if( ! (detcabl_it==detCabling_.end()) ){      // found detid in map
    return ( detcabl_it->second );
  }else{ // DKwarn : is there need for output message here telling det_id does not exist?
    static vector<FedChannelConnection> default_empty_fedchannelconnection;
    return default_empty_fedchannelconnection;
  }
}



const FedChannelConnection& SiStripDetCabling::getConnection( uint32_t det_id, unsigned short apv_pair ) const{
  const vector<FedChannelConnection>& fcconns = getConnections(det_id);
  for(vector<FedChannelConnection>::const_iterator iconn = fcconns.begin(); iconn!=fcconns.end();iconn++){
    if ( (iconn->apvPairNumber()) == apv_pair){ // check if apvPairNumber() of present FedChannelConnection is the same as requested one
      return (*iconn); // if yes, return the FedChannelConnection object
    }
  }
  // if did not match none of the above, return some default value - DKwarn : also output message?
  static FedChannelConnection default_empty_fedchannelconnection;
  return default_empty_fedchannelconnection;
}


const unsigned int SiStripDetCabling::getDcuId( uint32_t det_id ) const{
  const vector<FedChannelConnection>& fcconns = getConnections( det_id );
  if(! fcconns.size()==0) {
     return ( fcconns.at(1) ).dcuId(); // get dcuId of first element - when you build check this consistency
  }
  // default if none of the above is fulfilled
  unsigned int default_zero_value = 0;
  return default_zero_value;
}

