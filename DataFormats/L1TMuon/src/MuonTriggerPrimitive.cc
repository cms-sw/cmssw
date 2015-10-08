#include "DataFormats/L1TMuon/interface/MuonTriggerPrimitive.h"

// the primitive types we can use
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiL1Link.h"

// detector ID types
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

using namespace l1t;

namespace {
  const char subsystem_names[][4] = {"DT","CSC","RPC"};
}

//constructors from DT data
MuonTriggerPrimitive::MuonTriggerPrimitive(const DTChamberId& detid,
				   const L1MuDTChambPhDigi& digi_phi,
				   const int segment_number):
  _id(detid),
  _subsystem(MuonTriggerPrimitive::kDT) {
  calculateDTGlobalSector(detid,_globalsector,_subsector);
  // fill in information from theta trigger
  _dt.theta_bti_group = -1;
  _dt.segment_number = segment_number;  
  _dt.theta_code = -1;
  _dt.theta_quality = -1;
  // now phi trigger
  _dt.bx = digi_phi.bxNum();
  _dt.wheel = digi_phi.whNum();
  _dt.sector = digi_phi.scNum();
  _dt.station = digi_phi.stNum();
  _dt.radialAngle = digi_phi.phi();
  _dt.bendingAngle = digi_phi.phiB();
  _dt.qualityCode = digi_phi.code();
  _dt.Ts2TagCode = digi_phi.Ts2Tag();
  _dt.BxCntCode = digi_phi.BxCnt();
}

MuonTriggerPrimitive::MuonTriggerPrimitive(const DTChamberId& detid,
				   const L1MuDTChambThDigi& digi_th,
				   const int theta_bti_group):
  _id(detid),
  _subsystem(MuonTriggerPrimitive::kDT) {
  calculateDTGlobalSector(detid,_globalsector,_subsector);
  // fill in information from theta trigger
  _dt.theta_bti_group = theta_bti_group;
  _dt.segment_number = digi_th.position(theta_bti_group);
  _dt.theta_code = digi_th.code(theta_bti_group);
  _dt.theta_quality = digi_th.quality(theta_bti_group);
  // now phi trigger
  _dt.bx = digi_th.bxNum();
  _dt.wheel = digi_th.whNum();
  _dt.sector = digi_th.scNum();
  _dt.station = digi_th.stNum();
  _dt.radialAngle = -1;
  _dt.bendingAngle = -1;
  _dt.qualityCode = -1;
  _dt.Ts2TagCode = -1;
  _dt.BxCntCode = -1;
}

MuonTriggerPrimitive::MuonTriggerPrimitive(const DTChamberId& detid,
				   const L1MuDTChambPhDigi& digi_phi,
				   const L1MuDTChambThDigi& digi_th,
				   const int theta_bti_group):
  _id(detid),
  _subsystem(MuonTriggerPrimitive::kDT) {
  calculateDTGlobalSector(detid,_globalsector,_subsector);
  // fill in information from theta trigger
  _dt.theta_bti_group = theta_bti_group;
  _dt.segment_number = digi_th.position(theta_bti_group);
  _dt.theta_code = digi_th.code(theta_bti_group);
  _dt.theta_quality = digi_th.quality(theta_bti_group);
  // now phi trigger
  _dt.bx = digi_phi.bxNum();
  _dt.wheel = digi_phi.whNum();
  _dt.sector = digi_phi.scNum();
  _dt.station = digi_phi.stNum();
  _dt.radialAngle = digi_phi.phi();
  _dt.bendingAngle = digi_phi.phiB();
  _dt.qualityCode = digi_phi.code();
  _dt.Ts2TagCode = digi_phi.Ts2Tag();
  _dt.BxCntCode = digi_phi.BxCnt();    
}

//constructor from CSC data
MuonTriggerPrimitive::MuonTriggerPrimitive(const CSCDetId& detid,
				   const CSCCorrelatedLCTDigi& digi):
  _id(detid),
  _subsystem(MuonTriggerPrimitive::kCSC) {
  calculateCSCGlobalSector(detid,_globalsector,_subsector);
  _csc.trknmb  = digi.getTrknmb();
  _csc.valid   = digi.isValid();
  _csc.quality = digi.getQuality();
  _csc.keywire = digi.getKeyWG();
  _csc.strip   = digi.getStrip();
  _csc.pattern = digi.getPattern();
  _csc.bend    = digi.getBend();
  _csc.bx      = digi.getBX();
  _csc.mpclink = digi.getMPCLink();
  _csc.bx0     = digi.getBX0();
  _csc.syncErr = digi.getSyncErr();
  _csc.cscID   = digi.getCSCID();
}

// constructor from RPC data
MuonTriggerPrimitive::MuonTriggerPrimitive(const RPCDetId& detid,
				   const unsigned strip,
				   const unsigned layer,
				   const uint16_t bx):
  _id(detid),
  _subsystem(MuonTriggerPrimitive::kRPC) {
  calculateRPCGlobalSector(detid,_globalsector,_subsector);
  _rpc.strip = strip;
  _rpc.layer = layer;
  _rpc.bx = bx;
}

MuonTriggerPrimitive::MuonTriggerPrimitive(const MuonTriggerPrimitive& tp):
  _dt(tp._dt),
  _csc(tp._csc),
  _rpc(tp._rpc),
  _id(tp._id),
  _subsystem(tp._subsystem),  
  _globalsector(tp._globalsector),
  _subsector(tp._subsector),
  _eta(tp._eta),
  _phi(tp._phi),
  _theta(tp._theta){
}

MuonTriggerPrimitive& MuonTriggerPrimitive::operator=(const MuonTriggerPrimitive& tp) {
  this->_dt = tp._dt;
  this->_csc = tp._csc;
  this->_rpc = tp._rpc;
  this->_id = tp._id;
  this->_subsystem = tp._subsystem;
  this->_globalsector = tp._globalsector;
  this->_subsector = tp._subsector;
  this->_eta = tp._eta;
  this->_phi = tp._phi;
  return *this;
}

bool MuonTriggerPrimitive::operator==(const MuonTriggerPrimitive& tp) const {
  return ( this->_dt.bx == tp._dt.bx && 
	   this->_dt.wheel == tp._dt.wheel && 
	   this->_dt.sector == tp._dt.sector && 
	   this->_dt.station == tp._dt.station && 
	   this->_dt.radialAngle == tp._dt.radialAngle && 
	   this->_dt.bendingAngle == tp._dt.bendingAngle &&
	   this->_dt.qualityCode == tp._dt.qualityCode && 
	   this->_dt.Ts2TagCode == tp._dt.Ts2TagCode && 
	   this->_dt.BxCntCode == tp._dt.BxCntCode && 
	   this->_dt.theta_bti_group == tp._dt.theta_bti_group && 
	   this->_dt.segment_number == tp._dt.segment_number && 
	   this->_dt.theta_code == tp._dt.theta_code && 
	   this->_dt.theta_quality == tp._dt.theta_quality && 
	   this->_csc.trknmb == tp._csc.trknmb &&
	   this->_csc.valid == tp._csc.valid &&
	   this->_csc.quality == tp._csc.quality &&
	   this->_csc.keywire == tp._csc.keywire &&
	   this->_csc.strip == tp._csc.strip &&
	   this->_csc.pattern == tp._csc.pattern &&
	   this->_csc.bend == tp._csc.bend &&
	   this->_csc.bx == tp._csc.bx &&
	   this->_csc.mpclink == tp._csc.mpclink &&
	   this->_csc.bx0 == tp._csc.bx0 &&
	   this->_csc.syncErr == tp._csc.syncErr &&
	   this->_csc.cscID == tp._csc.cscID &&
	   this->_rpc.strip == tp._rpc.strip &&
	   this->_rpc.layer == tp._rpc.layer &&
	   this->_rpc.bx == tp._rpc.bx &&
	   this->_id == tp._id &&
	   this->_subsystem == tp._subsystem &&
	   this->_globalsector == tp._globalsector &&
	   this->_subsector == tp._subsector );	     
}

const int MuonTriggerPrimitive::getBX() const {
  switch(_subsystem) {
  case kDT:
    return _dt.bx;
  case kCSC:
    return _csc.bx;
  case kRPC:
    return _rpc.bx;
  default:
    throw cms::Exception("Invalid Subsytem") 
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
  return -1;
}

const int MuonTriggerPrimitive::getStrip() const {
  switch(_subsystem) {
  case kDT:
    return -1;
  case kCSC:
    return _csc.strip;
  case kRPC:
    return _rpc.strip;
  default:
    throw cms::Exception("Invalid Subsytem") 
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
  return -1;
}

const int MuonTriggerPrimitive::getWire() const {
  switch(_subsystem) {
  case kDT:
    return -1;
  case kCSC:
    return _csc.keywire;
  case kRPC:
    return -1;
  default:
    throw cms::Exception("Invalid Subsytem") 
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
  return -1;
}

const int MuonTriggerPrimitive::getPattern() const {
  switch(_subsystem) {
  case kDT:
    return -1;
  case kCSC:
    return _csc.pattern;
  case kRPC:
    return -1;
  default:
    throw cms::Exception("Invalid Subsytem") 
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
  return -1;
}
const int MuonTriggerPrimitive::Id() const {
  switch(_subsystem) {
  case kDT:
    return -1;
  case kCSC:
    return _csc.cscID;
  case kRPC:
    return -1;
  default:
    throw cms::Exception("Invalid Subsytem") 
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
  return -1;
}

void MuonTriggerPrimitive::calculateDTGlobalSector(const DTChamberId& chid, 
					       unsigned& global_sector, 
					       unsigned& subsector ) {
}

void MuonTriggerPrimitive::calculateCSCGlobalSector(const CSCDetId& chid, 
						unsigned& global_sector, 
						unsigned& subsector ) {
}

void MuonTriggerPrimitive::calculateRPCGlobalSector(const RPCDetId& chid, 
						unsigned& global_sector, 
						unsigned& subsector ) {
}

void MuonTriggerPrimitive::print(std::ostream& out) const {
  unsigned idx = (unsigned) _subsystem;
  out << subsystem_names[idx] << " Trigger Primitive" << std::endl;
  out << "eta: " << _eta << " phi: " << _phi 
      << " bend: " << _theta << std::endl;
  switch(_subsystem) {
  case kDT:
    out << detId<DTChamberId>() << std::endl;
    out << "Local BX      : " << _dt.bx << std::endl;
    out << "Segment Nmb   : " << _dt.segment_number << std::endl;
    out << "Packed Phi    : " << _dt.radialAngle << std::endl;
    out << "Packed Bend   : " << _dt.bendingAngle << std::endl;
    out << "Quality Code  : " << _dt.qualityCode << std::endl;
    out << "Ts2Tag Code   : " << _dt.Ts2TagCode << std::endl;
    out << "BXCnt Code    : " << _dt.BxCntCode << std::endl;
    out << "Theta BTI Grp : " << _dt.theta_bti_group << std::endl;
    out << "Theta Code    : " << _dt.theta_code << std::endl;
    out << "Theta Quality : " << _dt.theta_quality << std::endl;
    break;
  case kCSC:
    out << detId<CSCDetId>() << std::endl;
    out << "Local BX      : " << _csc.bx << std::endl;
    out << "Segment Nmb   : " << _csc.trknmb << std::endl;
    out << "Segment Valid : " << _csc.valid << std::endl;
    out << "Quality Code  : " << _csc.quality << std::endl;
    out << "Key Wire Grp  : " << _csc.keywire << std::endl;
    out << "Half-Strip    : " << _csc.strip << std::endl;
    out << "CLCT Pattern  : " << _csc.pattern << std::endl;
    out << "Packed Bend   : " << _csc.bend << std::endl;
    out << "MPC Link      : " << _csc.mpclink << std::endl;
    out << "BX0           : " << _csc.bx0 << std::endl;
    out << "Sync Error    : " << _csc.syncErr << std::endl;
    out << "CSCID         : " << _csc.cscID << std::endl;
    break;
  case kRPC:
    out << detId<RPCDetId>() << std::endl;
    out << "Local BX      : " << _rpc.bx << std::endl;
    out << "Strip         : " << _rpc.strip << std::endl;
    out << "Layer         : " << _rpc.layer << std::endl;
    break;
  default:
    throw cms::Exception("Invalid Subsytem") 
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }     
}
