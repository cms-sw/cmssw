#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
SiStripFecCabling::SiStripFecCabling( const SiStripFedCabling& cabling ) : fecs_() {
  cout << "[SiStripFecCabling::SiStripFecCabling]" 
       << " Constructing object..." << endl;
  
  const vector<unsigned short>& feds = cabling.feds();
  vector<unsigned short>::const_iterator ifed;
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
void SiStripFecCabling::countDevices( unsigned int& nfecs,
				      unsigned int& nrings,
				      unsigned int& nccus,
				      unsigned int& nmodules,
				      unsigned int& napvs,
				      unsigned int& ndcuids,
				      unsigned int& ndetids,
				      unsigned int& npairs,
				      unsigned int& nfedchans,
				      unsigned int& ndcus,
				      unsigned int& nmuxes,
				      unsigned int& nplls,
				      unsigned int& nllds ) {
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
	  napvs+= (*imod).apvs().size();
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

  // APVs
  vector<unsigned short>::iterator iapv;
  if ( conn.i2cAddrApv0() ) {
    iapv = find( apvs_.begin(), apvs_.end(), conn.i2cAddrApv0() );
    if ( iapv == apvs_.end() ) { apvs_.push_back( conn.i2cAddrApv0() ); }
  }
  if ( conn.i2cAddrApv1() ) {
    iapv = find( apvs_.begin(), apvs_.end(), conn.i2cAddrApv1() );
    if ( iapv == apvs_.end() ) { apvs_.push_back( conn.i2cAddrApv1() ); }
  }
  sort( apvs_.begin(), apvs_.end() );

  // Detector
  if ( !dcuId_ && conn.dcuId() )   { dcuId_ = conn.dcuId(); dcu0x00_ = true; }
  if ( !detId_ && conn.detId() )   { detId_ = conn.detId(); }
  if ( !nPairs_ && conn.nPairs() ) { nPairs_ = conn.nPairs(); }
  
  // FED cabling
  map< unsigned short, FedChannel>::iterator japv;
  japv = cabling_.find( conn.i2cAddrApv0() );
  if ( japv == cabling_.end() ) { 
    cabling_[conn.i2cAddrApv0()] = FedChannel(conn.fedId(),conn.fedCh());
  }
  japv = cabling_.find( conn.i2cAddrApv1() );
  if ( japv == cabling_.end() ) { 
    cabling_[conn.i2cAddrApv1()] = FedChannel(conn.fedId(),conn.fedCh());
  }
  
  // DCU, MUX, PLL, LLD
  if ( !dcu0x00_ && conn.dcu() ) { dcu0x00_ = true; }
  if ( !mux0x43_ && conn.mux() ) { mux0x43_ = true; }
  if ( !pll0x44_ && conn.pll() ) { pll0x44_ = true; }
  if ( !lld0x60_ && conn.lld() ) { lld0x60_ = true; }
  
}

// #include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
// EVENTSETUP_DATA_REG( SiStripFecCabling );
