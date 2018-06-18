#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"

// the primitive types we can use
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigi.h"

// detector ID types
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

using namespace L1TMuon;

namespace {
  const char subsystem_names[][4] = {"DT","CSC","RPC","GEM"};
}

//constructors from DT data
TriggerPrimitive::TriggerPrimitive(const DTChamberId& detid,
                                   const L1MuDTChambPhDigi& digi_phi,
                                   const int segment_number):
  _id(detid),
  _subsystem(TriggerPrimitive::kDT) {
  calculateGlobalSector(detid,_globalsector,_subsector);
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

TriggerPrimitive::TriggerPrimitive(const DTChamberId& detid,
                                   const L1MuDTChambThDigi& digi_th,
                                   const int theta_bti_group):
  _id(detid),
  _subsystem(TriggerPrimitive::kDT) {
  calculateGlobalSector(detid,_globalsector,_subsector);
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

TriggerPrimitive::TriggerPrimitive(const DTChamberId& detid,
                                   const L1MuDTChambPhDigi& digi_phi,
                                   const L1MuDTChambThDigi& digi_th,
                                   const int theta_bti_group):
  _id(detid),
  _subsystem(TriggerPrimitive::kDT) {
  calculateGlobalSector(detid,_globalsector,_subsector);
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
TriggerPrimitive::TriggerPrimitive(const CSCDetId& detid,
                                   const CSCCorrelatedLCTDigi& digi):
  _id(detid),
  _subsystem(TriggerPrimitive::kCSC) {
  calculateGlobalSector(detid,_globalsector,_subsector);
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

  // Use ME1/1a --> ring 4 convention
  if (detid.station() == 1 && detid.ring() == 1 && digi.getStrip() >= 128) {
    _id = CSCDetId(detid.endcap(), detid.station(), 4, detid.chamber(), detid.layer());
    _csc.strip = digi.getStrip() - 128;
  }

  CSCCorrelatedLCTDigi digi_clone = digi; // Necessary to get around const qualifier
  CSCALCTDigi alct = digi_clone.getALCT();
  CSCCLCTDigi clct = digi_clone.getCLCT();
  _csc.alct_quality = alct.getQuality();
  _csc.clct_quality = clct.getQuality();
}

// constructor from RPC data
TriggerPrimitive::TriggerPrimitive(const RPCDetId& detid,
                                   const RPCDigi& digi):
  _id(detid),
  _subsystem(TriggerPrimitive::kRPC) {
  calculateGlobalSector(detid,_globalsector,_subsector);
  _rpc.strip = digi.strip();
  _rpc.strip_low = digi.strip();
  _rpc.strip_hi = digi.strip();
  _rpc.layer = detid.layer();
  _rpc.bx = digi.bx();
  _rpc.valid = 1;
  _rpc.time = digi.time();
}

TriggerPrimitive::TriggerPrimitive(const RPCDetId& detid,
                                   const unsigned strip,
                                   const unsigned layer,
                                   const int bx):
  _id(detid),
  _subsystem(TriggerPrimitive::kRPC) {
  calculateGlobalSector(detid,_globalsector,_subsector);
  _rpc.strip = strip;
  _rpc.strip_low = strip;
  _rpc.strip_hi = strip;
  _rpc.layer = layer;
  _rpc.bx = bx;
  _rpc.valid = 1;
  _rpc.time = -999999.;
}


// constructor from GEM data
TriggerPrimitive::TriggerPrimitive(const GEMDetId& detid,
                                   const GEMPadDigi& digi):
  _id(detid),
  _subsystem(TriggerPrimitive::kGEM) {
  calculateGlobalSector(detid,_globalsector,_subsector);
  _gem.pad = digi.pad();
  _gem.pad_low = digi.pad();
  _gem.pad_hi = digi.pad();
  _gem.bx = digi.bx();
  _gem.bend = 0;
  _gem.isME0 = false;
}

TriggerPrimitive::TriggerPrimitive(const ME0DetId& detid,
                                   const ME0PadDigi& digi):
  _id(detid),
  _subsystem(TriggerPrimitive::kGEM) {
  calculateGlobalSector(detid,_globalsector,_subsector);
  _gem.pad = digi.pad();
  _gem.pad_low = digi.pad();
  _gem.pad_hi = digi.pad();
  _gem.bx = digi.bx();
  _gem.bend = 0;
  _gem.isME0 = true;
}

TriggerPrimitive::TriggerPrimitive(const TriggerPrimitive& tp):
  _dt(tp._dt),
  _csc(tp._csc),
  _rpc(tp._rpc),
  _gem(tp._gem),
  _id(tp._id),
  _subsystem(tp._subsystem),
  _globalsector(tp._globalsector),
  _subsector(tp._subsector),
  _eta(tp._eta),
  _phi(tp._phi),
  _rho(tp._rho),
  _theta(tp._theta){
}

TriggerPrimitive& TriggerPrimitive::operator=(const TriggerPrimitive& tp) {
  this->_dt = tp._dt;
  this->_csc = tp._csc;
  this->_rpc = tp._rpc;
  this->_gem = tp._gem;
  this->_id = tp._id;
  this->_subsystem = tp._subsystem;
  this->_globalsector = tp._globalsector;
  this->_subsector = tp._subsector;
  this->_eta = tp._eta;
  this->_phi = tp._phi;
  this->_rho = tp._rho;
  this->_theta = tp._theta;
  return *this;
}

bool TriggerPrimitive::operator==(const TriggerPrimitive& tp) const {
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
           this->_rpc.strip_low == tp._rpc.strip_low &&
           this->_rpc.strip_hi == tp._rpc.strip_hi &&
           this->_rpc.layer == tp._rpc.layer &&
           this->_rpc.bx == tp._rpc.bx &&
           this->_rpc.valid == tp._rpc.valid &&
           //this->_rpc.time == tp._rpc.time &&
           this->_gem.pad == tp._gem.pad &&
           this->_gem.pad_low == tp._gem.pad_low &&
           this->_gem.pad_hi == tp._gem.pad_hi &&
           this->_gem.bx == tp._gem.bx &&
           this->_gem.bend == tp._gem.bend &&
           this->_gem.isME0 == tp._gem.isME0 &&
           this->_id == tp._id &&
           this->_subsystem == tp._subsystem &&
           this->_globalsector == tp._globalsector &&
           this->_subsector == tp._subsector );
}

const int TriggerPrimitive::getBX() const {
  switch(_subsystem) {
  case kDT:
    return _dt.bx;
  case kCSC:
    return _csc.bx;
  case kRPC:
    return _rpc.bx;
  case kGEM:
    return _gem.bx;
  default:
    throw cms::Exception("Invalid Subsytem")
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
  return -1;
}

const int TriggerPrimitive::getStrip() const {
  switch(_subsystem) {
  case kDT:
    return -1;
  case kCSC:
    return _csc.strip;
  case kRPC:
    return _rpc.strip;
  case kGEM:
    return _gem.pad;
  default:
    throw cms::Exception("Invalid Subsytem")
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
  return -1;
}

const int TriggerPrimitive::getWire() const {
  switch(_subsystem) {
  case kDT:
    return -1;
  case kCSC:
    return _csc.keywire;
  case kRPC:
    return -1;
  case kGEM:
    return -1;
  default:
    throw cms::Exception("Invalid Subsytem")
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
  return -1;
}

const int TriggerPrimitive::getPattern() const {
  switch(_subsystem) {
  case kDT:
    return -1;
  case kCSC:
    return _csc.pattern;
  case kRPC:
    return -1;
  case kGEM:
    return -1;
  default:
    throw cms::Exception("Invalid Subsytem")
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
  return -1;
}

void TriggerPrimitive::print(std::ostream& out) const {
  unsigned idx = (unsigned) _subsystem;
  out << subsystem_names[idx] << " Trigger Primitive" << std::endl;
  out << "eta: " << _eta << " phi: " << _phi << " rho: " << _rho
      << " theta: " << _theta << std::endl;
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
    out << "Strip Low     : " << _rpc.strip_low << std::endl;
    out << "Strip High    : " << _rpc.strip_hi << std::endl;
    out << "Layer         : " << _rpc.layer << std::endl;
    out << "Valid         : " << _rpc.valid << std::endl;
    out << "Time          : " << _rpc.time << std::endl;
    break;
  case kGEM:
    if (!_gem.isME0)
      out << detId<GEMDetId>() << std::endl;
    else
      out << detId<ME0DetId>() << std::endl;
    out << "Local BX      : " << _gem.bx << std::endl;
    out << "Pad           : " << _gem.pad << std::endl;
    out << "Pad Low       : " << _gem.pad_low << std::endl;
    out << "Pad High      : " << _gem.pad_hi << std::endl;
    out << "Packed Bend   : " << _gem.bend << std::endl;
    out << "Is ME0        : " << _gem.isME0 << std::endl;
    break;
  default:
    throw cms::Exception("Invalid Subsytem")
      << "The specified subsystem for this track stub is out of range"
      << std::endl;
  }
}
