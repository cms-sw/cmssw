#include "L1Trigger/L1TMuon/interface/MuonTriggerPrimitive.h"

// Muon primitive digi types
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhDigi.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"

// Muon detector ID types
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"

#include <iostream>

using namespace L1TMuon;

namespace {
  const char subsystem_names[][4] = {"DT", "CSC", "RPC", "GEM", "ME0"};
}

// _____________________________________________________________________________
// Constructors from DT data
TriggerPrimitive::TriggerPrimitive(const DTChamberId& detid,
                                   const L1MuDTChambPhDigi& digi_phi,
                                   const int segment_number)
    : _id(detid), _subsystem(L1TMuon::kDT) {
  calculateGlobalSector(detid, _globalsector, _subsector);
  _eta = 0.;
  _phi = 0.;
  _rho = 0.;
  _theta = 0.;
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
  _dt.RpcBit = digi_phi.RpcBit();
}

TriggerPrimitive::TriggerPrimitive(const DTChamberId& detid,
                                   const L1MuDTChambThDigi& digi_th,
                                   const int theta_bti_group)
    : _id(detid), _subsystem(L1TMuon::kDT) {
  calculateGlobalSector(detid, _globalsector, _subsector);
  _eta = 0.;
  _phi = 0.;
  _rho = 0.;
  _theta = 0.;
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
  _dt.RpcBit = -10;
}

TriggerPrimitive::TriggerPrimitive(const DTChamberId& detid,
                                   const L1MuDTChambPhDigi& digi_phi,
                                   const L1MuDTChambThDigi& digi_th,
                                   const int theta_bti_group)
    : _id(detid), _subsystem(L1TMuon::kDT) {
  calculateGlobalSector(detid, _globalsector, _subsector);
  _eta = 0.;
  _phi = 0.;
  _rho = 0.;
  _theta = 0.;
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
  _dt.RpcBit = digi_phi.RpcBit();
}

// _____________________________________________________________________________
// Constructor from CSC data
TriggerPrimitive::TriggerPrimitive(const CSCDetId& detid, const CSCCorrelatedLCTDigi& digi)
    : _id(detid), _subsystem(L1TMuon::kCSC) {
  calculateGlobalSector(detid, _globalsector, _subsector);
  _eta = 0.;
  _phi = 0.;
  _rho = 0.;
  _theta = 0.;
  _csc.trknmb = digi.getTrknmb();
  _csc.valid = digi.isValid();
  _csc.quality = digi.getQuality();
  _csc.keywire = digi.getKeyWG();
  _csc.strip = digi.getStrip();
  _csc.pattern = digi.getPattern();
  _csc.bend = digi.getBend();
  _csc.bx = digi.getBX();
  _csc.mpclink = digi.getMPCLink();
  _csc.bx0 = digi.getBX0();
  _csc.syncErr = digi.getSyncErr();
  _csc.cscID = digi.getCSCID();
  _csc.alct_quality = digi.getALCT().getQuality();
  _csc.clct_quality = digi.getCLCT().getQuality();
  // run-3
  _csc.pattern_run3 = digi.getRun3Pattern();
  _csc.slope = digi.getSlope();
  _csc.strip_quart_bit = digi.getQuartStripBit();
  _csc.strip_eighth_bit = digi.getEightStripBit();
  _csc.strip_quart = digi.getStrip(4);
  _csc.strip_eighth = digi.getStrip(8);

  // Use ME1/1a --> ring 4 convention
  const bool is_me11a = (detid.station() == 1 && detid.ring() == 1 && digi.getStrip() >= 128);
  if (is_me11a) {
    _id = CSCDetId(detid.endcap(), detid.station(), 4, detid.chamber(), detid.layer());
    _csc.strip = digi.getStrip() - 128;
  }
}

// _____________________________________________________________________________
// Constructors from RPC data
TriggerPrimitive::TriggerPrimitive(const RPCDetId& detid, const RPCDigi& digi) : _id(detid), _subsystem(L1TMuon::kRPC) {
  calculateGlobalSector(detid, _globalsector, _subsector);
  _eta = 0.;
  _phi = 0.;
  _rho = 0.;
  _theta = 0.;
  _rpc.strip = digi.strip();
  _rpc.strip_low = digi.strip();
  _rpc.strip_hi = digi.strip();
  _rpc.phi_int = 0;
  _rpc.theta_int = 0;
  _rpc.emtf_sector = 0;
  _rpc.emtf_link = 0;
  _rpc.bx = digi.bx();
  _rpc.valid = 1;
  _rpc.x = digi.hasX() ? digi.coordinateX() : -999999.;
  _rpc.y = digi.hasY() ? digi.coordinateY() : -999999.;
  _rpc.time = digi.hasTime() ? digi.time() : -999999.;
  _rpc.isCPPF = false;
}

TriggerPrimitive::TriggerPrimitive(const RPCDetId& detid, const RPCRecHit& rechit)
    : _id(detid), _subsystem(L1TMuon::kRPC) {
  calculateGlobalSector(detid, _globalsector, _subsector);
  _eta = 0.;
  _phi = 0.;
  _rho = 0.;
  _theta = 0.;
  _rpc.strip = rechit.firstClusterStrip() + (rechit.clusterSize() - 1) / 2;
  _rpc.strip_low = rechit.firstClusterStrip();
  _rpc.strip_hi = rechit.firstClusterStrip() + rechit.clusterSize() - 1;
  _rpc.phi_int = 0;
  _rpc.theta_int = 0;
  _rpc.emtf_sector = 0;
  _rpc.emtf_link = 0;
  _rpc.bx = rechit.BunchX();
  _rpc.valid = 1;
  _rpc.x = rechit.localPosition().x();
  _rpc.y = rechit.localPosition().y();
  _rpc.time = rechit.time();
  _rpc.isCPPF = false;
}

// _____________________________________________________________________________
// Constructor from CPPF data
TriggerPrimitive::TriggerPrimitive(const RPCDetId& detid, const l1t::CPPFDigi& digi)
    : _id(detid), _subsystem(L1TMuon::kRPC) {
  calculateGlobalSector(detid, _globalsector, _subsector);
  _eta = 0.;
  _phi = 0.;
  _rho = 0.;
  _theta = 0.;
  // In unpacked CPPF digis, the strip number and cluster size are not available, and are set to -99
  _rpc.strip = (digi.first_strip() < 0 ? 0 : digi.first_strip() + ((digi.cluster_size() - 1) / 2));
  _rpc.strip_low = (digi.first_strip() < 0 ? 0 : digi.first_strip());
  _rpc.strip_hi = (digi.first_strip() < 0 ? 0 : digi.first_strip() + digi.cluster_size() - 1);
  _rpc.phi_int = digi.phi_int();
  _rpc.theta_int = digi.theta_int();
  _rpc.emtf_sector = digi.emtf_sector();
  _rpc.emtf_link = digi.emtf_link();
  _rpc.bx = digi.bx();
  _rpc.valid = digi.valid();
  _rpc.x = -999999.;
  _rpc.y = -999999.;
  _rpc.time = -999999.;
  _rpc.isCPPF = true;
}

// _____________________________________________________________________________
// Constructor from GEM data
TriggerPrimitive::TriggerPrimitive(const GEMDetId& detid, const GEMPadDigiCluster& digi)
    : _id(detid), _subsystem(L1TMuon::kGEM) {
  calculateGlobalSector(detid, _globalsector, _subsector);
  _eta = 0.;
  _phi = 0.;
  _rho = 0.;
  _theta = 0.;
  _gem.pad = digi.pads().front() + ((digi.pads().size() - 1) / 2);
  _gem.pad_low = digi.pads().front();
  _gem.pad_hi = digi.pads().front() + digi.pads().size() - 1;
  _gem.bx = digi.bx();
}

// _____________________________________________________________________________
// Constructor from ME0 data
TriggerPrimitive::TriggerPrimitive(const ME0DetId& detid, const ME0TriggerDigi& digi)
    : _id(detid), _subsystem(L1TMuon::kME0) {
  calculateGlobalSector(detid, _globalsector, _subsector);
  _eta = 0.;
  _phi = 0.;
  _rho = 0.;
  _theta = 0.;
  _me0.chamberid = digi.getChamberid();
  _me0.quality = digi.getQuality();
  _me0.phiposition = digi.getPhiposition();
  _me0.partition = digi.getPartition();
  _me0.deltaphi = digi.getDeltaphi();
  _me0.bend = digi.getBend();
  _me0.bx = digi.getBX();
}

// _____________________________________________________________________________
// Copy constructor
TriggerPrimitive::TriggerPrimitive(const TriggerPrimitive& tp)
    : _dt(tp._dt),
      _csc(tp._csc),
      _rpc(tp._rpc),
      _gem(tp._gem),
      _me0(tp._me0),
      _id(tp._id),
      _subsystem(tp._subsystem),
      _globalsector(tp._globalsector),
      _subsector(tp._subsector),
      _eta(tp._eta),
      _phi(tp._phi),
      _rho(tp._rho),
      _theta(tp._theta) {}

TriggerPrimitive& TriggerPrimitive::operator=(const TriggerPrimitive& tp) {
  this->_dt = tp._dt;
  this->_csc = tp._csc;
  this->_rpc = tp._rpc;
  this->_gem = tp._gem;
  this->_me0 = tp._me0;
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
  // Copied from Numpy
  // https://github.com/numpy/numpy/blob/v1.14.0/numpy/core/numeric.py#L2260-L2355
  auto isclose = [](float a, float b, float rtol = 1.e-5, float atol = 1.e-8) {
    return std::abs(a - b) <= (atol + rtol * std::abs(b));
  };

  switch (_subsystem) {
    case kDT:
      return (this->_dt.bx == tp._dt.bx && this->_dt.wheel == tp._dt.wheel && this->_dt.sector == tp._dt.sector &&
              this->_dt.station == tp._dt.station && this->_dt.radialAngle == tp._dt.radialAngle &&
              this->_dt.bendingAngle == tp._dt.bendingAngle && this->_dt.qualityCode == tp._dt.qualityCode &&
              this->_dt.Ts2TagCode == tp._dt.Ts2TagCode && this->_dt.BxCntCode == tp._dt.BxCntCode &&
              this->_dt.RpcBit == tp._dt.RpcBit && this->_dt.theta_bti_group == tp._dt.theta_bti_group &&
              this->_dt.segment_number == tp._dt.segment_number && this->_dt.theta_code == tp._dt.theta_code &&
              this->_dt.theta_quality == tp._dt.theta_quality && this->_id == tp._id &&
              this->_subsystem == tp._subsystem && this->_globalsector == tp._globalsector &&
              this->_subsector == tp._subsector);
    case kCSC:
      return (this->_csc.trknmb == tp._csc.trknmb && this->_csc.valid == tp._csc.valid &&
              this->_csc.quality == tp._csc.quality && this->_csc.keywire == tp._csc.keywire &&
              this->_csc.strip == tp._csc.strip && this->_csc.pattern == tp._csc.pattern &&
              this->_csc.bend == tp._csc.bend && this->_csc.bx == tp._csc.bx && this->_csc.mpclink == tp._csc.mpclink &&
              this->_csc.bx0 == tp._csc.bx0 && this->_csc.syncErr == tp._csc.syncErr &&
              this->_csc.cscID == tp._csc.cscID && this->_csc.alct_quality == tp._csc.alct_quality &&
              this->_csc.clct_quality == tp._csc.clct_quality && this->_id == tp._id &&
              this->_subsystem == tp._subsystem && this->_globalsector == tp._globalsector &&
              this->_subsector == tp._subsector);
    case kRPC:
      return (this->_rpc.strip == tp._rpc.strip && this->_rpc.strip_low == tp._rpc.strip_low &&
              this->_rpc.strip_hi == tp._rpc.strip_hi && this->_rpc.phi_int == tp._rpc.phi_int &&
              this->_rpc.theta_int == tp._rpc.theta_int && this->_rpc.emtf_sector == tp._rpc.emtf_sector &&
              this->_rpc.emtf_link == tp._rpc.emtf_link && this->_rpc.bx == tp._rpc.bx &&
              this->_rpc.valid == tp._rpc.valid && isclose(this->_rpc.x, tp._rpc.x) &&  // floating-point
              isclose(this->_rpc.y, tp._rpc.y) &&                                       // floating-point
              isclose(this->_rpc.time, tp._rpc.time) &&                                 // floating-point
              this->_rpc.isCPPF == tp._rpc.isCPPF && this->_id == tp._id && this->_subsystem == tp._subsystem &&
              this->_globalsector == tp._globalsector && this->_subsector == tp._subsector);
    case kGEM:
      return (this->_gem.pad == tp._gem.pad && this->_gem.pad_low == tp._gem.pad_low &&
              this->_gem.pad_hi == tp._gem.pad_hi && this->_gem.bx == tp._gem.bx && this->_id == tp._id &&
              this->_subsystem == tp._subsystem && this->_globalsector == tp._globalsector &&
              this->_subsector == tp._subsector);
    case kME0:
      return (this->_me0.chamberid == tp._me0.chamberid && this->_me0.quality == tp._me0.quality &&
              this->_me0.phiposition == tp._me0.phiposition && this->_me0.partition == tp._me0.partition &&
              this->_me0.deltaphi == tp._me0.deltaphi && this->_me0.bend == tp._me0.bend &&
              this->_me0.bx == tp._me0.bx && this->_id == tp._id && this->_subsystem == tp._subsystem &&
              this->_globalsector == tp._globalsector && this->_subsector == tp._subsector);
    default:
      throw cms::Exception("Invalid Subsystem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
  return false;
}

// _____________________________________________________________________________
int TriggerPrimitive::getBX() const {
  switch (_subsystem) {
    case kDT:
      return _dt.bx;
    case kCSC:
      return _csc.bx;
    case kRPC:
      return _rpc.bx;
    case kGEM:
      return _gem.bx;
    case kME0:
      return _me0.bx;
    default:
      throw cms::Exception("Invalid Subsystem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
  return -1;
}

int TriggerPrimitive::getStrip() const {
  switch (_subsystem) {
    case kDT:
      return _dt.radialAngle;
    case kCSC:
      return _csc.strip;
    case kRPC:
      return _rpc.strip;
    case kGEM:
      return _gem.pad;
    case kME0:
      return _me0.phiposition;
    default:
      throw cms::Exception("Invalid Subsystem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
  return -1;
}

int TriggerPrimitive::getWire() const {
  switch (_subsystem) {
    case kDT:
      return _dt.theta_bti_group;
    case kCSC:
      return _csc.keywire;
    case kRPC:
      return -1;
    case kGEM:
      return -1;
    case kME0:
      return -1;
    default:
      throw cms::Exception("Invalid Subsystem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
  return -1;
}

int TriggerPrimitive::getPattern() const {
  switch (_subsystem) {
    case kDT:
      return -1;
    case kCSC:
      return _csc.pattern;
    case kRPC:
      return -1;
    case kGEM:
      return -1;
    case kME0:
      return -1;
    default:
      throw cms::Exception("Invalid Subsystem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
  return -1;
}

void TriggerPrimitive::print(std::ostream& out) const {
  unsigned idx = (unsigned)_subsystem;
  out << subsystem_names[idx] << " Trigger Primitive" << std::endl;
  out << "eta: " << _eta << " phi: " << _phi << " rho: " << _rho << " theta: " << _theta << std::endl;
  switch (_subsystem) {
    case kDT:
      out << detId<DTChamberId>() << std::endl;
      out << "Local BX      : " << _dt.bx << std::endl;
      out << "Segment Nmb   : " << _dt.segment_number << std::endl;
      out << "Packed Phi    : " << _dt.radialAngle << std::endl;
      out << "Packed Bend   : " << _dt.bendingAngle << std::endl;
      out << "Quality Code  : " << _dt.qualityCode << std::endl;
      out << "Ts2Tag Code   : " << _dt.Ts2TagCode << std::endl;
      out << "BxCnt Code    : " << _dt.BxCntCode << std::endl;
      out << "RPC Bit       : " << _dt.RpcBit << std::endl;
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
      out << "ALCT Quality  : " << _csc.alct_quality << std::endl;
      out << "CLCT Quality  : " << _csc.clct_quality << std::endl;
      break;
    case kRPC:
      out << detId<RPCDetId>() << std::endl;
      out << "Local BX      : " << _rpc.bx << std::endl;
      out << "Strip         : " << _rpc.strip << std::endl;
      out << "Strip low     : " << _rpc.strip_low << std::endl;
      out << "Strip high    : " << _rpc.strip_hi << std::endl;
      out << "Integer phi   : " << _rpc.phi_int << std::endl;
      out << "Integer theta : " << _rpc.theta_int << std::endl;
      out << "EMTF sector   : " << _rpc.emtf_sector << std::endl;
      out << "EMTF link     : " << _rpc.emtf_link << std::endl;
      out << "Valid         : " << _rpc.valid << std::endl;
      out << "Local x       : " << _rpc.x << std::endl;
      out << "Local y       : " << _rpc.y << std::endl;
      out << "Time          : " << _rpc.time << std::endl;
      out << "IsCPPF        : " << _rpc.isCPPF << std::endl;
      break;
    case kGEM:
      out << detId<GEMDetId>() << std::endl;
      out << "Local BX      : " << _gem.bx << std::endl;
      out << "Pad           : " << _gem.pad << std::endl;
      out << "Pad low       : " << _gem.pad_low << std::endl;
      out << "Pad high      : " << _gem.pad_hi << std::endl;
      break;
    case kME0:
      out << detId<ME0DetId>() << std::endl;
      out << "Local BX      : " << _me0.bx << std::endl;
      out << "Chamber id    : " << _me0.chamberid << std::endl;
      out << "Quality       : " << _me0.quality << std::endl;
      out << "Phiposition   : " << _me0.phiposition << std::endl;
      out << "Partition     : " << _me0.partition << std::endl;
      out << "Deltaphi      : " << _me0.deltaphi << std::endl;
      out << "Bend          : " << _me0.bend << std::endl;
      break;
    default:
      throw cms::Exception("Invalid Subsystem")
          << "The specified subsystem for this track stub is out of range" << std::endl;
  }
}
