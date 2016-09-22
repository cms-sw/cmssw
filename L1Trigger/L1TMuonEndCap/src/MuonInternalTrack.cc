#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrack.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackCand.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1Track.h"

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"

using namespace L1TMuon;

InternalTrack::InternalTrack(const L1MuDTTrackCand& dttrack):
  L1MuRegionalCand(dttrack) {
  _mode = 0;
  _wheel = dttrack.whNum();
  _endcap = (_wheel < 0) ? -1 : 1;
  _sector = dttrack.scNum()/2 + 1; // 0-11 -> 1 - 6  
}

InternalTrack::InternalTrack(const csc::L1Track& csctrack):
  L1MuRegionalCand(csctrack) {
  _mode = 0;
  _endcap = (csctrack.endcap() == 2) ? -1 : 1;
  _wheel = (_endcap < 0) ? -4 : 4;
  _sector = csctrack.sector();
}

InternalTrack::InternalTrack(const L1MuRegionalCand& rpctrack,
			     const RPCL1LinkRef& rpclink):
  L1MuRegionalCand(rpctrack) {
  _parentlink = rpclink;
  _mode = 0;
  _endcap = -99;
  _wheel  = -99;
  _sector = -99;
}

unsigned InternalTrack::type_idx() const {
  if( _parent.isNonnull() ) return L1MuRegionalCand::type_idx();
  return _type;
}

void InternalTrack::addStub(const TriggerPrimitive& stub) { 
  unsigned station;
  subsystem_offset offset;
  TriggerPrimitive::subsystem_type type = stub.subsystem();
  switch(type){
  case TriggerPrimitive::kCSC:    
    offset = kCSC;
    station = stub.detId<CSCDetId>().station();
    break;
  case TriggerPrimitive::kDT:    
    offset = kDT;
    station = stub.detId<DTChamberId>().station();
    break;
  case TriggerPrimitive::kRPC:    
    offset = kRPCb;
    if(stub.detId<RPCDetId>().region() != 0) 
      offset = kRPCf;
    station = stub.detId<RPCDetId>().station(); 
    break;
  default:
    throw cms::Exception("Invalid Subsytem") 
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }  

  const unsigned shift = 4*offset + station - 1;
  const unsigned bit = 1 << shift;
   // add this track to the mode
  _mode = _mode | bit;
  if( _associatedStubs.count(shift) == 0 ) {
    _associatedStubs[shift] = TriggerPrimitiveCollection();
  }   
  _associatedStubs[shift].push_back(stub);
}

// this magic file contains a DT TrackClass -> mode LUT
#include "L1Trigger/DTTrackFinder/src/L1MuDTTrackAssParam.h"

void InternalTrack::print(std::ostream& out) const {
  std::cout << "Internal Track -- endcap: " << std::dec << _endcap
	    << " wheel: " << _wheel 
	    << " sector: " << _sector << std::endl;
  std::cout << "\t eta index: " << eta_packed() 
	    << "\t phi index: " << phi_packed() << std::endl;
  std::cout << "\tMode: " << std::hex  
	    << mode() << std::dec << std::endl;
  std::cout << "\tMode Breakdown: " << std::hex
	    << " DT: " << dtMode() << " RPCb: " << rpcbMode()
	    << " CSC: " << cscMode() << " RPCf: " << rpcfMode() 
	    << std::dec << std::endl;
  std::cout << "\tQuality: " << quality() << std::endl;
  DTTrackRef dtparent;
  CSCTrackRef cscparent;
  RegionalCandRef rpcparent;
  unsigned mode;
  switch( type_idx() ) {
  case 0: // DT    
    dtparent = _parent.castTo<DTTrackRef>();
    mode = tc2bitmap((TrackClass)dtparent->TCNum());
    std::cout << "\tParent is a DT Track!" << std::endl;
    std::cout << "\t Parent Track Class: " << dtparent->TCNum() << std::endl;
    std::cout << "\t Parent Mode: " << std::hex
	      << mode
	      << std::dec << std::endl;
    std::cout << "\t  MB 1: " << dtparent->stNum(1)
	      << "\t  MB 2: " << dtparent->stNum(2)
	      << "\t  MB 3: " << dtparent->stNum(3)
	      << "\t  MB 4: " << dtparent->stNum(4) << std::endl;
    if( (mode & 0x1) != (dtMode() & 0x1) ) {
      std::cout << "DT-Based Internal Track did not find expected DT"
		<< " segment in station 1!" << std::endl;
    }
    if( (mode & 0x2) != (dtMode() & 0x2) ) {
      std::cout << "DT-Based Internal Track did not find expected DT"
		<< " segment in station 2!" << std::endl;
    }
    if( std::abs(dtparent->whNum()) == 3 ) {
      if( dtparent->stNum(3) == 0 || dtparent->stNum(3) == 1) { // CSC track!
	if( (mode & 0x4) != ((cscMode() & 0x1)<<2) ) {
	  std::cout << "DT-Based Internal Track did not find expected CSC"
		    << " segment in station 1!" << std::endl;
	}
      } else {
	if( (mode & 0x4) != (dtMode() & 0x4) ) {
	  std::cout << "DT-Based Internal Track did not find expected DT"
		    << " segment in station 3!" << std::endl;
	}
      }
    } else {
      if( (mode & 0x4) != (dtMode() & 0x4) ) {
	std::cout << "DT-Based Internal Track did not find expected DT"
		  << " segment in station 3!" << std::endl;
      }
    }
    if(	(mode & 0x8) != (dtMode() & 0x8) ) {
      std::cout << "DT-Based Internal Track did not find expected DT"
		<< " segment in station 4!" << std::endl;
    }
    std::cout << "\t Parent Quality: " << dtparent->quality() << std::endl;
    break;
  case 1: // RPCb
    rpcparent = _parent.castTo<RegionalCandRef>();
    std::cout << "\tParent is a RPCb Track!" << std::endl;
    std::cout << "\t Parent Quality: " << rpcparent->quality() << std::endl;
    std::cout << "\t Parent phi: " << rpcparent->phi_packed() << std::endl;
    std::cout << "\t Parent eta: " << rpcparent->eta_packed() << std::endl;
    break;
  case 2: // CSC    
    cscparent = _parent.castTo<CSCTrackRef>();
    std::cout << "\tParent is a CSC Track!" << std::endl;
    std::cout << "\t Parent Mode: " << std::hex
	      << cscparent->mode() 
	      << std::dec << std::endl
	      << "\t  ME 1: " << cscparent->me1ID() 
	      << "\t  ME 2: " << cscparent->me2ID() 
	      << "\t  ME 3: " << cscparent->me3ID() 
	      << "\t  ME 4: " << cscparent->me4ID() 
	      << "\t  MB 1: " << cscparent->mb1ID() << std::endl;
    if( (bool)(cscparent->me1ID()) != (bool)(cscMode() & 0x1) ) {
      std::cout << "CSC-Based Internal Track did not find expected CSC"
		<< " segment in station 1!" << std::endl;
    }
    if( (bool)(cscparent->me2ID()) != (bool)(cscMode() & 0x2) ) {
      std::cout << "CSC-Based Internal Track did not find expected CSC"
		<< " segment in station 2!" << std::endl;
    }
    if(	(bool)(cscparent->me3ID()) != (bool)(cscMode() & 0x4) ) {
      std::cout << "CSC-Based Internal Track did not find expected CSC"
		<< " segment in station 3!" << std::endl;
    }
    if(	(bool)(cscparent->me4ID()) != (bool)(cscMode() & 0x8) ) {
      std::cout << "CSC-Based Internal Track did not find expected CSC"
		<< " segment in station 4!" << std::endl;
    }
    if(	(bool)(cscparent->mb1ID()) != (bool)(dtMode() & 0x1)  ) {
      std::cout << "CSC-Based Internal Track did not find expected DT"
		<< " segment in station 1!" << std::endl;
    }
    std::cout << "\t Parent Quality: " << cscparent->quality() << std::endl;
    break;
  case 3: // RPCf
    rpcparent = _parent.castTo<RegionalCandRef>();
    std::cout << "\tParent is a RPCf Track!" << std::endl;
    std::cout << "\t Parent Quality: " << rpcparent->quality() << std::endl;
    std::cout << "\t Parent phi: " << rpcparent->phi_packed() << std::endl;
    std::cout << "\t Parent eta: " << rpcparent->eta_packed() << std::endl;
    break;
  case 4: // L1ITMu ?
    break;
  default:
    throw cms::Exception("Unknown Track Type") 
      << "L1ITMu::InternalTrack is of unknown track type: " << type_idx()
      << std::endl;
  }
}
