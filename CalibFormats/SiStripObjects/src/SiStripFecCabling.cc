#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
SiStripFecCabling::SiStripFecCabling( const SiStripFedCabling& cabling ) : fecs_() {
  cout << "[SiStripFecCabling::SiStripFecCabling]" 
       << " Constructing object..." << endl;
  
  const vector<uint16_t>& feds = cabling.feds();
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    const vector<FedChannelConnection>& conns = cabling.connections( *ifed ); 
    vector<FedChannelConnection>::const_iterator iconn;
    for ( iconn = conns.begin(); iconn != conns.end(); iconn++ ) {
      addDevices( *iconn );
    }
  }
  
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::connections( vector<FedChannelConnection>& conns ) {
  cout << "[SiStripFecCabling::connections]" << endl;
  conns.clear();
  const vector<SiStripFec>& fecs = (*this).fecs();
  for ( vector<SiStripFec>::const_iterator ifec = fecs.begin(); ifec != fecs.end(); ifec++ ) {
    const vector<SiStripRing>& rings = (*ifec).rings();
    for ( vector<SiStripRing>::const_iterator iring = rings.begin(); iring != rings.end(); iring++ ) {
      const vector<SiStripCcu>& ccus = (*iring).ccus();
      for ( vector<SiStripCcu>::const_iterator iccu = ccus.begin(); iccu != ccus.end(); iccu++ ) {
	const vector<SiStripModule>& modules = (*iccu).modules();
	for ( vector<SiStripModule>::const_iterator imod = modules.begin(); imod != modules.end(); imod++ ) {
	  
	  for ( uint16_t ipair = 0; ipair < (*imod).nPairs(); ipair++ ) {
	    
	    


// 	    FedChannelConnection( (*imod).fec_crate, 
// 				  fec_slot, 
// 				  fec_ring, 
// 				  ccu_addr, 
// 				  ccu_chan, 
// 				  apv0 = 0,
// 				apv1 = 0,
// 				uint32_t dcu_id = 0,
// 				uint32_t det_id = 0,
// 				pairs  = 0,
// 				fed_id = 0,
// 				fed_ch = 0,
// 				length = 0,
// 				dcu = false,
// 				pll = false,
// 				mux = false,
// 				lld = false
//  )

// 	  conns.push_back( (*imod). );
	  } 
	}
      }
    }
  }
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::addDevices( const FedChannelConnection& conn ) {
  cout << "[SiStripFecCabling::addDevices]" << endl;
  vector<SiStripFec>::const_iterator ifec = fecs().begin();
  while ( ifec != fecs().end() && (*ifec).fecSlot() != conn.fecSlot() ) { ifec++; }
  if ( ifec == fecs().end() ) { fecs_.push_back( SiStripFec( conn ) ); }
  else { const_cast<SiStripFec&>(*ifec).addDevices( conn ); }
}

// -----------------------------------------------------------------------------
//
SiStripModule& SiStripFecCabling::module( const FedChannelConnection& conn ) {
  cout << "[SiStripFecCabling::module]" << endl;
  
  vector<SiStripFec>::const_iterator ifec = fecs().begin();
  while ( ifec != fecs().end() && (*ifec).fecSlot() != conn.fecSlot() ) { ifec++; }
  if ( ifec != fecs().end() ) { 
    vector<SiStripRing>::const_iterator iring = (*ifec).rings().begin();
    while ( iring != (*ifec).rings().end() && (*iring).fecRing() != conn.fecRing() ) { iring++; }
    if ( iring != (*ifec).rings().end() ) { 
      vector<SiStripCcu>::const_iterator iccu = (*iring).ccus().begin();
      while ( iccu != (*iring).ccus().end() && (*iccu).ccuAddr() != conn.ccuAddr() ) { iccu++; }
      if ( iccu != (*iring).ccus().end() ) { 
	vector<SiStripModule>::const_iterator imod = (*iccu).modules().begin();
	while ( imod != (*iccu).modules().end() && (*imod).ccuChan() != conn.ccuChan() ) { imod++; }
	if ( imod != (*iccu).modules().end() ) { 
	  return const_cast<SiStripModule&>( *imod );
	} else { cerr << "[SiStripFecCabling::module]"
		      << " CCU channel " << conn.ccuChan() 
		      << " not found!" << endl; }
      } else { cerr << "[SiStripFecCabling::module]"
		    << " CCU address " << conn.ccuAddr() 
		    << " not found!" << endl; }
    } else { cerr << "[SiStripFecCabling::module]"
		  << " FEC ring " << conn.fecRing() 
		  << " not found!" << endl; }
  } else { cerr << "[SiStripFecCabling::module]"
		<< " FEC slot " << conn.fecSlot() 
		<< " not found!" << endl; }
  static FedChannelConnection temp;
  static SiStripModule module(temp);
  return module;
}

// -----------------------------------------------------------------------------
//
void SiStripFecCabling::countDevices( uint32_t& nfecs,
				      uint32_t& nrings,
				      uint32_t& nccus,
				      uint32_t& nmodules,
				      uint32_t& napvs,
				      uint32_t& ndcuids,
				      uint32_t& ndetids,
				      uint32_t& npairs,
				      uint32_t& nfedchans,
				      uint32_t& ndcus,
				      uint32_t& nmuxes,
				      uint32_t& nplls,
				      uint32_t& nllds ) {
  cout << "[SiStripFecCabling::countDevices]" << endl;
  
  nfecs = nrings = nccus = nmodules = napvs = 0;
  ndcuids = ndetids = npairs = nfedchans = 0;
  ndcus = nmuxes = nplls = nllds = 0;
  const vector<SiStripFec>& fecs = (*this).fecs();
  for ( vector<SiStripFec>::const_iterator ifec = fecs.begin(); ifec != fecs.end(); ifec++ ) {
    nfecs++;
    const vector<SiStripRing>& rings = (*ifec).rings();
    for ( vector<SiStripRing>::const_iterator iring = rings.begin(); iring != rings.end(); iring++ ) {
      nrings++;
      const vector<SiStripCcu>& ccus = (*iring).ccus();
      for ( vector<SiStripCcu>::const_iterator iccu = ccus.begin(); iccu != ccus.end(); iccu++ ) {
	nccus++;
	const vector<SiStripModule>& modules = (*iccu).modules();
	for ( vector<SiStripModule>::const_iterator imod = modules.begin(); imod != modules.end(); imod++ ) {
	  nmodules++;
	  if ( (*imod).apv(32) ) { napvs++; }
	  if ( (*imod).apv(33) ) { napvs++; }
	  if ( (*imod).apv(34) ) { napvs++; }
	  if ( (*imod).apv(35) ) { napvs++; }
	  if ( (*imod).apv(36) ) { napvs++; }
	  if ( (*imod).apv(37) ) { napvs++; }
	  if ( (*imod).dcuId() ) { ndcuids++; }
	  if ( (*imod).detId() ) { ndetids++; }
	  npairs += (*imod).nPairs();
	  // fedchans++
	  if ( (*imod).dcu() ) { ndcus++; }
	  if ( (*imod).mux() ) { nmuxes++; }
	  if ( (*imod).pll() ) { nplls++; }
	  if ( (*imod).lld() ) { nllds++; }
	} 
      }
    }
  }
  
}

// -----------------------------------------------------------------------------
//
void SiStripFec::addDevices( const FedChannelConnection& conn ) {
  cout << "[SiStripFec::addDevices]" << endl;
  vector<SiStripRing>::const_iterator iring = rings().begin();
  while ( iring != rings().end() && (*iring).fecRing() != conn.fecRing() ) { iring++; }
  if ( iring == rings().end() ) { rings_.push_back( SiStripRing( conn ) ); }
  else { const_cast<SiStripRing&>(*iring).addDevices( conn ); }
}

// -----------------------------------------------------------------------------
//
void SiStripRing::addDevices( const FedChannelConnection& conn ) {
  cout << "[SiStripRing::addDevices]" << endl;
  vector<SiStripCcu>::const_iterator iccu = ccus().begin();
  while ( iccu != ccus().end() && (*iccu).ccuAddr() != conn.ccuAddr() ) { iccu++; }
  if ( iccu == ccus().end() ) { ccus_.push_back( SiStripCcu( conn ) ); }
  else { const_cast<SiStripCcu&>(*iccu).addDevices( conn ); }
}

// -----------------------------------------------------------------------------
//
void SiStripCcu::addDevices( const FedChannelConnection& conn ) {
  cout << "[SiStripCcu::addDevices]" << endl;
  vector<SiStripModule>::const_iterator imod = modules().begin();
  while ( imod != modules().end() && (*imod).ccuChan() != conn.ccuChan() ) { imod++; }
  if ( imod == modules().end() ) { modules_.push_back( SiStripModule( conn ) ); }
  else { const_cast<SiStripModule&>(*imod).addDevices( conn ); }
}

// -----------------------------------------------------------------------------
//
void SiStripModule::addDevices( const FedChannelConnection& conn ) {
  cout << "[SiStripModule::addDevices]" << endl;

  //@@ NEED CHECKS HERE!!!

  // APV0
  if ( conn.i2cAddrApv0() == 32 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv0() == 33 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv0() == 34 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv0() == 35 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv0() == 36 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv0() == 37 ) { apv0x32_ = true; }

  // APV1
  if ( conn.i2cAddrApv1() == 32 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv1() == 33 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv1() == 34 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv1() == 35 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv1() == 36 ) { apv0x32_ = true; }
  if ( conn.i2cAddrApv1() == 37 ) { apv0x32_ = true; }
  
  // Detector
  if ( !dcuId_ && conn.dcuId() )   { dcuId_ = conn.dcuId(); dcu0x00_ = true; }
  if ( !detId_ && conn.detId() )   { detId_ = conn.detId(); }
  if ( !nPairs_ && conn.nPairs() ) { nPairs_ = conn.nPairs(); }
  
  // FED cabling
  map< uint16_t, pair<uint16_t,uint16_t> >::iterator japv;
  japv = cabling_.find( conn.i2cAddrApv0() );
  if ( japv == cabling_.end() ) { 
    cabling_[conn.i2cAddrApv0()] = pair<uint16_t,uint16_t>( conn.fedId(), conn.fedCh() );
  }
  japv = cabling_.find( conn.i2cAddrApv1() );
  if ( japv == cabling_.end() ) { 
    cabling_[conn.i2cAddrApv1()] = pair<uint16_t,uint16_t>( conn.fedId(), conn.fedCh() );
  }
  
  // DCU, MUX, PLL, LLD
  if ( !dcu0x00_ && conn.dcu() ) { dcu0x00_ = true; }
  if ( !mux0x43_ && conn.mux() ) { mux0x43_ = true; }
  if ( !pll0x44_ && conn.pll() ) { pll0x44_ = true; }
  if ( !lld0x60_ && conn.lld() ) { lld0x60_ = true; }
  
}

// -----------------------------------------------------------------------------
//
vector<uint16_t> SiStripModule::apvs() {
  cout << "[SiStripFecCabling::apvs]" << endl;
  vector<uint16_t> apvs;
  if ( apv0x32_ ) { apvs.push_back( apv0x32_ ); }
  if ( apv0x33_ ) { apvs.push_back( apv0x33_ ); }
  if ( apv0x34_ ) { apvs.push_back( apv0x34_ ); }
  if ( apv0x35_ ) { apvs.push_back( apv0x35_ ); }
  if ( apv0x36_ ) { apvs.push_back( apv0x36_ ); }
  if ( apv0x37_ ) { apvs.push_back( apv0x37_ ); }
  return apvs;
}

// -----------------------------------------------------------------------------
//
uint16_t SiStripModule::apv( uint16_t apv_id ) const {
  cout << "[SiStripFecCabling::apv]" << endl;
  if      ( apv_id == 0 || apv_id == 32 ) { return apv0x32_; }
  else if ( apv_id == 1 || apv_id == 33 ) { return apv0x33_; }
  else if ( apv_id == 2 || apv_id == 34 ) { return apv0x34_; }
  else if ( apv_id == 3 || apv_id == 35 ) { return apv0x35_; }
  else if ( apv_id == 4 || apv_id == 36 ) { return apv0x36_; }
  else if ( apv_id == 5 || apv_id == 37 ) { return apv0x37_; }
  else {
    cerr << "[SiStripFecCabling::apv]" 
	 << " Unexpected APV number!" << endl;
  }
  return uint16_t(0);
}

// // -----------------------------------------------------------------------------
// //
// pair<uint16_t,uint16_t>  SiStripModule::pair( uint16_t apv_pair ) const {
//   cout << "[SiStripFecCabling::pair]" << endl;
//   if ( nPairs_ == 2 ) {
//     if      ( apv_pair == 0 ) { return pair<uint16_t,uint16_t>(apv0x32_,apv0x33_); }
//     else if ( apv_pair == 1 ) { return pair<uint16_t,uint16_t>(apv0x36_,apv0x37_); }
//     else                      { return pair<uint16_t,uint16_t>(0,0); }
//   } else if ( nPairs_ == 3 ) {
//     if      ( apv_pair == 0 ) { return pair<uint16_t,uint16_t>(apv0x32_,apv0x33_); }
//     else if ( apv_pair == 1 ) { return pair<uint16_t,uint16_t>(apv0x34_,apv0x35_); }
//     else if ( apv_pair == 2 ) { return pair<uint16_t,uint16_t>(apv0x36_,apv0x37_); }
//     else                      { return pair<uint16_t,uint16_t>(0,0); }
//   } else {
//     cerr << "[SiStripFecCabling::pair]" << " Unexpected number of pairs!" << endl;
//   }
// }

// // -----------------------------------------------------------------------------
// //
// pair<uint16_t,uint16_t>  SiStripModule::fedCh( uint16_t apv_pair ) const {
//   cout << "[SiStripFecCabling::fedCh]" << endl;
//   map< uint16_t, pair<uint16_t,uint16_t> >::iterator ipair;
//   ipair = find( cabling_.begin(), cabling_.end(), apv_pair );
//   if ( ipair != cabling_.end() ) { 
//     return pair<uint16_t,uint16_t>(cabling_[ipair].first,cabling_[ipair].second);
//   } else {
//     return pair<uint16_t,uint16_t>(0,0);
//   }
// }

// #include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
// EVENTSETUP_DATA_REG( SiStripFecCabling );
