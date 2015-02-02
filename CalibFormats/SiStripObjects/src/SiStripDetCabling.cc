// -*- C++ -*-
// Package:     SiStripObjects
// Class  :     SiStripDetCabling
// Original Author:  dkcira
//         Created:  Wed Mar 22 12:24:33 CET 2006
#include "FWCore/Utilities/interface/typelookup.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include <iostream>

//---- default constructor / destructor
SiStripDetCabling::SiStripDetCabling() : fedCabling_(0) {}
SiStripDetCabling::~SiStripDetCabling() {}

//---- construct detector view (DetCabling) out of readout view (FedCabling)
SiStripDetCabling::SiStripDetCabling(const SiStripFedCabling& fedcabling) : fullcabling_(), connected_(), detected_(), undetected_(), fedCabling_(&fedcabling)
{
  // --- CONNECTED = have fedid and i2cAddr
  // create fullcabling_, loop over vector of FedChannelConnection, either make new element of map, or add to appropriate vector of existing map element
  // get feds list (vector) from fedcabling object - these are the active FEDs
  auto feds = fedcabling.fedIds();
  for ( auto ifed = feds.begin(); ifed != feds.end(); ifed++ ) { // iterate over active feds, get all their FedChannelConnection-s
    SiStripFedCabling::ConnsConstIterRange conns = fedcabling.fedConnections( *ifed );
    for ( auto iconn = conns.begin(); iconn != conns.end(); iconn++ ) { // loop over FedChannelConnection objects
      addDevices(*iconn, fullcabling_); // leave separate method, in case you will need to add devices also after constructing
      bool have_fed_id = iconn->fedId();
      std::vector<int> vector_of_connected_apvs;
      if(have_fed_id){ // these apvpairs are seen from the readout
	// there can be at most 6 APVs on one DetId: 0,1,2,3,4,5
        int which_apv_pair = iconn->apvPairNumber(); // APVPair (0,1) for 512 strips and (0,1,2) for 768 strips
	
	// patch needed to take into account invalid detids or apvPairs
	if( iconn->detId()==0 ||  
	    iconn->detId() == sistrip::invalid32_ ||  
	    iconn->apvPairNumber() == sistrip::invalid_  ||
	    iconn->nApvPairs() == sistrip::invalid_ ) {
	  continue;
	}

	if(iconn->i2cAddr(0)) vector_of_connected_apvs.push_back(2*which_apv_pair + 0); // first apv of the pair
	if(iconn->i2cAddr(1)) vector_of_connected_apvs.push_back(2*which_apv_pair + 1); // second apv of the pair
      }
      if(vector_of_connected_apvs.size() != 0){ // add only is smth. there, obviously
        std::map<uint32_t, std::vector<int> > map_of_connected_apvs;
        map_of_connected_apvs.insert(std::make_pair(iconn->detId(),vector_of_connected_apvs));
        addFromSpecificConnection(connected_, map_of_connected_apvs, &(connectionCount[0]));
      }
    }
  }
  // --- DETECTED = do not have fedid but have i2cAddr
  SiStripFedCabling::ConnsConstIterRange detected_fed_connections = fedcabling.detectedDevices();
  for(std::vector<FedChannelConnection>::const_iterator idtct = detected_fed_connections.begin(); idtct != detected_fed_connections.end(); idtct++){
    addDevices(*idtct, fullcabling_);
    bool have_fed_id = idtct->fedId();
    std::vector<int> vector_of_detected_apvs;
    if(! have_fed_id){
      int which_apv_pair = idtct->apvPairNumber(); // APVPair (0,1) for 512 strips and (0,1,2) for 768 strips
      if(idtct->i2cAddr(0)) vector_of_detected_apvs.push_back(2*which_apv_pair + 0); // first apv of the pair
      if(idtct->i2cAddr(1)) vector_of_detected_apvs.push_back(2*which_apv_pair + 1); // second apv of the pair
    }
    if(vector_of_detected_apvs.size() != 0){ // add only is smth. there, obviously
      std::map<uint32_t,std::vector<int> > map_of_detected_apvs;
      map_of_detected_apvs.insert(std::make_pair(idtct->detId(),vector_of_detected_apvs));
      addFromSpecificConnection(detected_, map_of_detected_apvs, &(connectionCount[1]) );
    }
  }
  // --- UNDETECTED = have neither fedid nor i2caddr
  SiStripFedCabling::ConnsConstIterRange undetected_fed_connections = fedcabling.undetectedDevices();
  for(std::vector<FedChannelConnection>::const_iterator iudtct = undetected_fed_connections.begin(); iudtct != undetected_fed_connections.end(); iudtct++){
    addDevices(*iudtct, fullcabling_);
    bool have_fed_id = iudtct->fedId();
    std::vector<int> vector_of_undetected_apvs;
    if(! have_fed_id){
      int which_apv_pair = iudtct->apvPairNumber(); // APVPair (0,1) for 512 strips and (0,1,2) for 768 strips
      if(iudtct->i2cAddr(0)) vector_of_undetected_apvs.push_back(2*which_apv_pair + 0); // first apv of the pair
      if(iudtct->i2cAddr(1)) vector_of_undetected_apvs.push_back(2*which_apv_pair + 1); // second apv of the pair
    }
    if(vector_of_undetected_apvs.size() != 0){ // add only is smth. there, obviously
      std::map<uint32_t, std::vector<int> > map_of_undetected_apvs;
      map_of_undetected_apvs.insert(std::make_pair(iudtct->detId(),vector_of_undetected_apvs));
      addFromSpecificConnection(undetected_, map_of_undetected_apvs, &(connectionCount[2]));
    }
  }
}

//---- add to certain connections
void SiStripDetCabling::addDevices( const FedChannelConnection& conn, 
                                    std::map< uint32_t, std::vector<const FedChannelConnection *> >& conns ){
  if( conn.detId() && conn.detId() != sistrip::invalid32_ &&  // check for valid detid
      conn.apvPairNumber() != sistrip::invalid_ ) {           // check for valid apv pair number
    if( conn.fedId()==0 || conn.fedId()==sistrip::invalid_ ){
      edm::LogInfo("") << " SiStripDetCabling::addDevices for connection associated to detid " << conn.detId() << " apvPairNumber " << conn.apvPairNumber() << "the fedId is " << conn.fedId();
      return;
    }
    // check cached vector size is sufficient
    // if not, resize
    if( conn.apvPairNumber() >= conns[conn.detId()].size() ) {
      conns[conn.detId()].resize( conn.apvPairNumber()+1 );
    }
    // add latest connection object
    conns[conn.detId()][conn.apvPairNumber()] = &conn;
  }
}

//----
void SiStripDetCabling::addDevices(const FedChannelConnection & conn){ // by default add to fullcabling_ connections - special case of above class
  addDevices(conn, fullcabling_ ); // add to fullcabling_
}

//---- get vector of connected modules. replaces getActiveDetectorRawIds method - avoid use of static
void SiStripDetCabling::addActiveDetectorsRawIds(std::vector<uint32_t> & vector_to_fill_with_detids ) const{
  for(std::map< uint32_t, std::vector<int> >::const_iterator conn_it = connected_.begin(); conn_it!=connected_.end(); conn_it++){
    vector_to_fill_with_detids.push_back(conn_it->first);
  }
  // no elements added to vector_to_fill_with_detids is empty connected_
}

//---- get vector of all modules.
void SiStripDetCabling::addAllDetectorsRawIds(std::vector<uint32_t> & vector_to_fill_with_detids ) const{
  for(std::map< uint32_t, std::vector<int> >::const_iterator conn_it = connected_.begin(); conn_it!=connected_.end(); conn_it++){
    vector_to_fill_with_detids.push_back(conn_it->first);
  }
  for(std::map< uint32_t, std::vector<int> >::const_iterator conn_it = detected_.begin(); conn_it!=detected_.end(); conn_it++){
    vector_to_fill_with_detids.push_back(conn_it->first);
  }
  for(std::map< uint32_t, std::vector<int> >::const_iterator conn_it = undetected_.begin(); conn_it!=undetected_.end(); conn_it++){
    vector_to_fill_with_detids.push_back(conn_it->first);
  }
  // no elements added to vector_to_fill_with_detids is empty connected_, detected_.begin and undetected_.begin
}

//----
const std::vector<const FedChannelConnection *>& SiStripDetCabling::getConnections(uint32_t det_id ) const{ // return all connections corresponding to one det_id
  std::map< uint32_t, std::vector<const FedChannelConnection *> >::const_iterator detcabl_it = fullcabling_.find(det_id); // has to be const_iterator because this function cannot change data members
  if( ! (detcabl_it==fullcabling_.end()) ){  // found detid in fullcabling_
    return ( detcabl_it->second );
  }else{ // DKwarn : is there need for output message here telling det_id does not exist?
    const static std::vector<const FedChannelConnection *> default_empty_fedchannelconnection;
    return default_empty_fedchannelconnection;
  }
}

//----
const FedChannelConnection& SiStripDetCabling::getConnection( uint32_t det_id, unsigned short apv_pair ) const{
  const std::vector<const FedChannelConnection *>& fcconns = getConnections(det_id);
  for(std::vector<const FedChannelConnection *>::const_iterator iconn = fcconns.begin(); iconn!=fcconns.end();iconn++){
    if ( ((*iconn) != 0) && (((*iconn)->apvPairNumber()) == apv_pair) ) { // check if apvPairNumber() of present FedChannelConnection is the same as requested one
      return (**iconn); // if yes, return the FedChannelConnection object
    }
  }
  // if did not match none of the above, return some default value - DKwarn : also output message?
  const static FedChannelConnection default_empty_fedchannelconnection;
  return default_empty_fedchannelconnection;
}

//----
const unsigned int SiStripDetCabling::getDcuId( uint32_t det_id ) const{
  const std::vector<const FedChannelConnection *>& fcconns = getConnections( det_id );
  if(fcconns.size()!=0) {
    // patch needed to take into account the possibility that the first component of fcconns is invalid
    for(size_t i=0;i<fcconns.size();++i)
      if (fcconns.at(i) && fcconns.at(i)->detId() != sistrip::invalid32_ && fcconns.at(i)->detId() != 0 )
        return ( fcconns.at(i) )->dcuId(); // get dcuId of first element - when you build check this consistency
  }
  // default if none of the above is fulfilled
  unsigned int default_zero_value = 0;
  return default_zero_value;
}

//---- one can find the nr of apvs from fullcabling_ -> std::vector<FedChannelConnection> -> size * 2
const uint16_t SiStripDetCabling::nApvPairs(uint32_t det_id) const{
  const std::vector<const FedChannelConnection *>& fcconns = getConnections( det_id );
  if(fcconns.size()!=0) {
    // patch needed to take into account the possibility that the first component of fcconns is invalid
    for(size_t i=0;i<fcconns.size();++i) {
      if ( (fcconns.at(i) != 0) && (fcconns.at(i)->nApvPairs() != sistrip::invalid_) ) {
        return fcconns.at(i)->nApvPairs(); // nr of apvpairs for associated module
      }
    }
  }
  // else {
  //   return 0;
  // }
  return 0;
}

//---- map of detector to list of APVs for APVs seen from FECs and FEDs
void SiStripDetCabling::addConnected ( std::map<uint32_t, std::vector<int> > & map_to_add_to) const{
  addFromSpecificConnection(map_to_add_to, connected_);
}

//--- map of detector to list of APVs for APVs seen neither from FECS or FEDs
void SiStripDetCabling::addDetected( std::map<uint32_t, std::vector<int> > & map_to_add_to) const{
  addFromSpecificConnection(map_to_add_to, detected_);
}

//---- map of detector to list of APVs for APVs seen neither from FECS or FEDs
void SiStripDetCabling::addUnDetected( std::map<uint32_t, std::vector<int> > & map_to_add_to) const{
  addFromSpecificConnection(map_to_add_to, undetected_);
}

//----  map of detector to list of APVs that are not connected - combination of addDetected and addUnDetected
void SiStripDetCabling::addNotConnectedAPVs( std::map<uint32_t, std::vector<int> > & map_to_add_to) const{
  addFromSpecificConnection(map_to_add_to, detected_);
  addFromSpecificConnection(map_to_add_to, undetected_);
}

//----
void SiStripDetCabling::addFromSpecificConnection( std::map<uint32_t, std::vector<int> > & map_to_add_to,
                                                   const std::map< uint32_t, std::vector<int> > & specific_connection,
                                                   std::map< int16_t, uint32_t >* connectionsToFill ) const {
  for(std::map< uint32_t, std::vector<int> >::const_iterator conn_it = specific_connection.begin(); conn_it!=specific_connection.end(); ++conn_it){
    uint32_t new_detid = conn_it->first;
    std::vector<int> new_apv_vector = conn_it->second;
    std::map<uint32_t, std::vector<int> >::iterator it = map_to_add_to.find(new_detid);
    if( it == map_to_add_to.end() ){ // detid does not exist in map, add new entry
      std::sort(new_apv_vector.begin(),new_apv_vector.end()); // not very efficient sort, time consuming?
      map_to_add_to.insert(std::make_pair(new_detid,new_apv_vector));

      // Count the number of detIds per layer. Doing it in this "if" we count each detId only once
      // (otherwise it would be counted once per APV pair)
      // ATTENTION: consider changing the loop content to avoid this
      // This is the expected full number of modules (double sided are counted twice because the two
      // sides have different detId).
      // TIB1 : 336, TIB2 : 432, TIB3 : 540, TIB4 : 648
      // TID : each disk has 48+48+40 (ring1+ring2+ring3)
      // TOB1 : 504, TOB2 : 576, TOB3 : 648, TOB4 : 720, TOB5 : 792, TOB6 : 888
      // TEC1 : Total number of modules = 6400.
      if( connectionsToFill ) {
        (*connectionsToFill)[layerSearch(new_detid)]++;
      }

    }else{                    // detid exists already, add to its vector - if its not there already . . .
      std::vector<int> existing_apv_vector = it->second;
      for(std::vector<int>::iterator inew = new_apv_vector.begin(); inew != new_apv_vector.end(); inew++ ){
        bool there_already = false;
        for(std::vector<int>::iterator iold = existing_apv_vector.begin(); iold != existing_apv_vector.end(); iold++){
	  if (*iold == *inew){
	    there_already = true;
	    break; // leave the loop
	  }
        }
        if( ! there_already ){
          existing_apv_vector.push_back(*inew);
          std::sort(existing_apv_vector.begin(),existing_apv_vector.end()); // not very efficient sort, time consuming?
        }else{
          //edm::LogWarning("Logical") << "apv "<<*inew<<" already exists in the detector module "<<new_detid;
        }
      }
    }
  }
}

int16_t SiStripDetCabling::layerSearch( const uint32_t detId ) const
{
  if(SiStripDetId(detId).subDetector()==SiStripDetId::TIB){
    TIBDetId D(detId);
    return D.layerNumber();
  } else if (SiStripDetId(detId).subDetector()==SiStripDetId::TID){
    TIDDetId D(detId);
    // side: 1 = negative, 2 = positive
    return 10+(D.side() -1)*3 + D.wheel();
  } else if (SiStripDetId(detId).subDetector()==SiStripDetId::TOB){
    TOBDetId D(detId);
    return 100+D.layerNumber();
  } else if (SiStripDetId(detId).subDetector()==SiStripDetId::TEC){
    TECDetId D(detId);
    // side: 1 = negative, 2 = positive
    return 1000+(D.side() -1)*9 + D.wheel();
  }
  return 0;
}

/// Return the number of modules for the specified subDet, layer and connectionType.
uint32_t SiStripDetCabling::detNumber(const std::string & subDet, const uint16_t layer, const int connectionType) const {
  uint16_t subDetLayer = layer;
  // TIB = 1, TID = 2, TOB = 3, TEC = 4
  if( subDet == "TID-" ) subDetLayer += 10;
  else if( subDet == "TID+" ) subDetLayer += 10 + 3;
  else if( subDet == "TOB" ) subDetLayer += 100;
  else if( subDet == "TEC-" ) subDetLayer += 1000;
  else if( subDet == "TEC+" ) subDetLayer += 1000 + 9;
  else if( subDet != "TIB" ) {
    LogDebug("SiStripDetCabling") << "Error: Wrong subDet. Please use one of TIB, TID, TOB, TEC." << std::endl;
    return 0;
  }
  auto found = connectionCount[connectionType].find(subDetLayer);
  if(found != connectionCount[connectionType].end()) {
    return found->second;
  }
  return 0;
}

//---- map of all connected, detected, undetected to contiguous Ids - map reset first!
void SiStripDetCabling::getAllDetectorsContiguousIds(std::map<uint32_t, unsigned int>& allToContiguous) const{
  allToContiguous.clear(); // reset map
  std::vector<uint32_t> all; addAllDetectorsRawIds(all); std::sort(all.begin(), all.end()); // get all detids and sort them
  unsigned int contiguousIndex = 0;
  for(std::vector<uint32_t>::const_iterator idet = all.begin(); idet!= all.end(); ++idet){
     ++contiguousIndex;
     allToContiguous.insert(std::make_pair(*idet,contiguousIndex)); 
  }
}

//---- map of all connected - map reset first!
void SiStripDetCabling::getActiveDetectorsContiguousIds(std::map<uint32_t, unsigned int>& connectedToContiguous) const{
  connectedToContiguous.clear(); // reset map
  std::vector<uint32_t> connected; addAllDetectorsRawIds(connected); std::sort(connected.begin(), connected.end()); // get connected detids and sort them (not strictly necessary)
  std::map<uint32_t, unsigned int> allToContiguous; getAllDetectorsContiguousIds(allToContiguous); // create map of all indices
  for(std::vector<uint32_t>::const_iterator idet = connected.begin(); idet!= connected.end(); ++idet){ // select only the indices for active detectors
    std::map<uint32_t, unsigned int>::iterator deco = allToContiguous.find(*idet);
    if(deco!=allToContiguous.end()){
       connectedToContiguous.insert(*deco);
    }
  }
}

bool SiStripDetCabling::IsConnected(const uint32_t& det_id) const {
  return IsInMap(det_id,connected_);
}

bool SiStripDetCabling::IsDetected(const uint32_t& det_id) const {
  return IsInMap(det_id,detected_);
}
bool SiStripDetCabling::IsUndetected(const uint32_t& det_id) const{
  return IsInMap(det_id,undetected_);
}
bool SiStripDetCabling::IsInMap(const uint32_t& det_id, const std::map<uint32_t, std::vector<int> > & map) const{
  std::map< uint32_t, std::vector<int> >::const_iterator it=map.find(det_id);
  return (it!=map.end());
}

// -----------------------------------------------------------------------------
/** Added missing print method. */
void SiStripDetCabling::print( std::stringstream& ss ) const {
  uint32_t valid = 0;
  uint32_t total = 0;
  typedef std::vector<const FedChannelConnection *> Conns;
  typedef std::map<uint32_t,Conns> ConnsMap;
  ConnsMap::const_iterator ii = fullcabling_.begin();
  ConnsMap::const_iterator jj = fullcabling_.end();
  ss << "[SiStripDetCabling::" << __func__ << "]"
     << " Printing DET cabling for " << fullcabling_.size()
     << " modules " << std::endl;
  for ( ; ii != jj; ++ii ) {
    ss << "Printing " << ii->second.size()
       << " connections for DetId: " << ii->first << std::endl;
    Conns::const_iterator iii = ii->second.begin();
    Conns::const_iterator jjj = ii->second.end();
    for ( ; iii != jjj; ++iii ) { 
      if ( (*iii)->isConnected() ) { valid++; }
      total++;
      ss << **iii << std::endl;
    }
  }
  ss << "Number of connected:   " << valid << std::endl
     << "Number of connections: " << total << std::endl;
}

void SiStripDetCabling::printSummary(std::stringstream& ss) const {
  for( int connectionType = 0; connectionType < 3; ++connectionType ) {
    if( connectionType == 0 ) ss << "Connected modules:" << std::endl;
    else if( connectionType == 1 ) ss << "Detected modules:" << std::endl;
    else ss << "Undetected modules:" << std::endl;
    ss << "SubDet and layer\t modules" << std::endl;
    std::map< int16_t, uint32_t >::const_iterator iter = connectionCount[connectionType].begin();
    for( ; iter != connectionCount[connectionType].end(); ++iter ) {
      uint32_t subDetLayer = iter->first;
      uint32_t modules = iter->second;
      if( int(subDetLayer/10) == 0 ) {
        ss << "TIB \t layer " << subDetLayer << " \t" << modules << std::endl;
      }
      else if( int(subDetLayer/100) == 0 ) {
        int layer = subDetLayer%10;
        if( layer <= 3 ) ss << "TID- \t disk  " << layer << "\t" << modules << std::endl;
        else ss << "TID+ \t disk  " << layer-3 << "\t" << modules << std::endl;
      }
      else if( int(subDetLayer/1000) == 0 ) {
        int layer = subDetLayer%100;
        ss << "TOB \t layer " << layer << " \t" << modules << std::endl;
      }
      else {
        int layer = subDetLayer%100;
        if( layer <= 9 ) ss << "TEC- \t disk  " << layer << " \t" << modules << std::endl;
        else ss << "TEC+ \t disk  " << layer-9 << " \t" << modules << std::endl;
      }
    }
  }
}

void SiStripDetCabling::printDebug(std::stringstream& ss) const {
  print(ss);
}
