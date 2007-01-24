// -*- C++ -*-
//
// Package:     SiStripObjects
// Class  :     SiStripDetCabling
// Implementation:
//     <Notes on implementation>
// Original Author:  dkcira
//         Created:  Wed Mar 22 12:24:33 CET 2006
// $Id: SiStripDetCabling.cc,v 1.3 2006/05/15 08:51:23 dkcira Exp $

#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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


SiStripDetCabling::SiStripDetCabling(const SiStripFedCabling& fedcabling) : connected_(), detected_(), undetected_()
{
 // --- CONNECTED
 // create connected_, loop over vector of FedChannelConnection, either make new element of map, or add to appropriate vector of existing map element
  // get feds list (vector) from fedcabling object - these are the active FEDs
  const vector<uint16_t>& feds = fedcabling.feds();
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) { // iterate over active feds, get all their FedChannelConnection-s
    const vector<FedChannelConnection>& conns = fedcabling.connections( *ifed );
    vector<FedChannelConnection>::const_iterator iconn;
    for ( iconn = conns.begin(); iconn != conns.end(); iconn++ ) { // loop over FedChannelConnection objects
      addDevices(*iconn, connected_); // leave separate method, in case you will need to add devices also after constructing
    }
  }
 // --- DETECTED
 const std::vector<FedChannelConnection>& detected_fed_connections = fedcabling.detected();
 for(std::vector<FedChannelConnection>::const_iterator idtct = detected_fed_connections.begin(); idtct != detected_fed_connections.end(); idtct++){
      addDevices(*idtct, detected_);
 }
 // --- UNDETECTED
 const std::vector<FedChannelConnection>& undetected_fed_connections = fedcabling.undetected();
 for(std::vector<FedChannelConnection>::const_iterator iudtct = undetected_fed_connections.begin(); iudtct != undetected_fed_connections.end(); iudtct++){
      addDevices(*iudtct, undetected_);
 }
}

void SiStripDetCabling::addDevices(const FedChannelConnection & conn, std::map< uint32_t, std::vector<FedChannelConnection> >& chosen_connections_ ){ // add to certain connections
// separate method in case need to add devices/cabling after object already constructed
  const uint32_t& fedchannelconnection_detid = conn.detId(); // detId of FedChannelConnection that is looped over
  // is this detid already in the map?
  map< uint32_t, vector<FedChannelConnection> >::iterator detcabl_it = chosen_connections_.find(fedchannelconnection_detid);
  if( ! (detcabl_it==chosen_connections_.end()) ){      // is already in map -- DKwarn : add constistency checks here for example agreement with already existing dcuId?
    ( detcabl_it->second ).push_back(conn); // add one FedChannelConnection to vector mapped to this detid
  }else{                                      // is not in map, create vector with 1 element and put into map
    vector<FedChannelConnection> local_fedchannelconnection; local_fedchannelconnection.push_back(conn);
    chosen_connections_[fedchannelconnection_detid] = local_fedchannelconnection;
  }
}


void SiStripDetCabling::addDevices(const FedChannelConnection & conn){ // by default add to connected_ connections - special case of above class
 addDevices(conn, connected_ ); // add to connected_
}

void SiStripDetCabling::addActiveDetectorsRawIds(std::vector<uint32_t> & vector_to_fill_with_detids ) const{ // replace getActiveDetectorRawIds method - avoid use of static
 for(map< uint32_t, vector<FedChannelConnection> >::const_iterator detcabl_it = connected_.begin(); detcabl_it!=connected_.end(); detcabl_it++){
   vector_to_fill_with_detids.push_back(detcabl_it->first);
 }
 // vector_to_fill_with_detids is empty if no elements in connected_
}


void SiStripDetCabling::addConnectedDetectorsRawIds(std::vector<uint32_t> & vector_to_fill_with_detids) const{ // wrapper around addActiveDetectorsRawIds
  addActiveDetectorsRawIds(vector_to_fill_with_detids);
}


void SiStripDetCabling::addDetectedDetectorsRawIds(std::vector<uint32_t> & vector_to_fill_with_detids ) const{
 for(map< uint32_t, vector<FedChannelConnection> >::const_iterator detcabl_it = detected_.begin(); detcabl_it!=detected_.end(); detcabl_it++){
   vector_to_fill_with_detids.push_back(detcabl_it->first);
 }
}

void SiStripDetCabling::addUnDetectedDetectorsRawIds(std::vector<uint32_t> & vector_to_fill_with_detids ) const{
 for(map< uint32_t, vector<FedChannelConnection> >::const_iterator detcabl_it = undetected_.begin(); detcabl_it!=undetected_.end(); detcabl_it++){
   vector_to_fill_with_detids.push_back(detcabl_it->first);
 }
}


const vector<FedChannelConnection>& SiStripDetCabling::getConnections(uint32_t det_id ) const{ // return all connections corresponding to one det_id
  map< uint32_t, vector<FedChannelConnection> >::const_iterator detcabl_it1 = connected_.find(det_id); // has to be const_iterator because this function cannot change data members
  map< uint32_t, vector<FedChannelConnection> >::const_iterator detcabl_it2 = detected_.find(det_id);
  map< uint32_t, vector<FedChannelConnection> >::const_iterator detcabl_it3 = undetected_.find(det_id);
  if( ! (detcabl_it1==connected_.end()) ){  // found detid in connected_
    return ( detcabl_it1->second );
  }
  if( ! (detcabl_it2==detected_.end()) ){   // found detid in detected_
    return ( detcabl_it2->second );
  }
  if( ! (detcabl_it3==undetected_.end()) ){ // found detid in detected_
    return ( detcabl_it3->second );
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

EVENTSETUP_DATA_REG(SiStripDetCabling);
