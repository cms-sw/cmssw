#ifndef __L1TMuon_TriggerPrimitive_h__
#define __L1TMuon_TriggerPrimitive_h__
//
// Class: L1TMuon::TriggerPrimitive
//
// Info: This class implements a unifying layer between DT, CSC and RPC
//       trigger primitives (TPs) such that TPs from different subsystems
//       can be queried for their position and orientation information
//       in a consistent way amongst all subsystems.
//
// Note: Not all input data types are persistable, so we make local
//       copies of all data from various digi types.
//
//       At the end of the day this should represent the output of some
//       common sector receiver module.
//
// Author: L. Gray (FNAL)
//

#include <cstdint>
#include <vector>
#include <iosfwd>

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/L1TMuon/interface/L1TMuonSubsystems.h"

// DT digi types
class DTChamberId;
class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;

// CSC digi types
class CSCCorrelatedLCTDigi;
class CSCDetId;

// RPC digi types
class RPCRecHit;
class RPCDigi;
class RPCDetId;

// CPPF digi types
namespace l1t {
  class CPPFDigi;
}

// GEM digi types
class GEMPadDigiCluster;
class GEMDetId;

// ME0 digi types
class ME0TriggerDigi;
class ME0DetId;

namespace L1TMuon {

  class TriggerPrimitive {
  public:
    // define the data we save locally from each subsystem type
    // variables in these structs keep their colloquial meaning
    // within a subsystem
    // for RPCs you have to unroll the digi-link and raw det-id
    struct RPCData {
      RPCData()
          : strip(0),
            strip_low(0),
            strip_hi(0),
            phi_int(0),
            theta_int(0),
            emtf_sector(0),
            emtf_link(0),
            bx(0),
            valid(0),
            x(0.),
            y(0.),
            time(0.),
            isCPPF(false) {}
      uint16_t strip;
      uint16_t strip_low;    // for use in clustering
      uint16_t strip_hi;     // for use in clustering
      uint16_t phi_int;      // for CPPFDigis in EMTF
      uint16_t theta_int;    // for CPPFDigis in EMTF
      uint16_t emtf_sector;  // for CPPFDigis in EMTF
      uint16_t emtf_link;    // for CPPFDigis in EMTF
      int16_t bx;
      int16_t valid;
      float x;     // local coordinate x (use floating-point for now)
      float y;     // local coordinate y (use floating-point for now)
      float time;  // time (use floating-point for now)
      bool isCPPF;
    };

    struct CSCData {
      CSCData()
          : trknmb(0),
            valid(0),
            quality(0),
            keywire(0),
            strip(0),
            pattern(0),
            bend(0),
            bx(0),
            mpclink(0),
            bx0(0),
            syncErr(0),
            cscID(0),
            alct_quality(0),
            clct_quality(0),
            // run-3
            pattern_run3(0),
            strip_quart_bit(0),
            strip_eighth_bit(0),
            strip_quart(0),
            strip_eighth(0),
            slope(0) {}
      uint16_t trknmb;
      uint16_t valid;
      uint16_t quality;
      uint16_t keywire;
      uint16_t strip;
      uint16_t pattern;
      uint16_t bend;
      uint16_t bx;
      uint16_t mpclink;
      uint16_t bx0;
      uint16_t syncErr;
      uint16_t cscID;
      uint16_t alct_quality;  // extra info for ALCT (wires)
      uint16_t clct_quality;  // extra info for CLCT (strips)
      // run-3
      uint16_t pattern_run3;
      uint16_t strip_quart_bit;
      uint16_t strip_eighth_bit;
      uint16_t strip_quart;
      uint16_t strip_eighth;
      uint16_t slope;
    };

    struct DTData {
      DTData()
          : bx(0),
            wheel(0),
            sector(0),
            station(0),
            radialAngle(0),
            bendingAngle(0),
            qualityCode(0),
            Ts2TagCode(0),
            BxCntCode(0),
            RpcBit(-10),
            theta_bti_group(0),
            segment_number(0),
            theta_code(0),
            theta_quality(0) {}
      // from ChambPhDigi (corresponds to a TRACO)
      // this gives us directly the phi
      int bx;            // relative? bx number
      int wheel;         // wheel number -3,-2,-1,1,2,3
      int sector;        // 1-12 in DT speak (these correspond to CSC sub-sectors)
      int station;       // 1-4 radially outwards
      int radialAngle;   // packed phi in a sector
      int bendingAngle;  // angle of segment relative to chamber
      int qualityCode;   // need to decode
      int Ts2TagCode;    // ??
      int BxCntCode;     // ????
      int RpcBit;        // 0: DT only, 1: DT segment BX corrected by RPC, 2: RPC only
      // from ChambThDigi (corresponds to a BTI)
      // we have to root out the eta manually
      // theta super layer == SL 1
      // station four has no theta super-layer
      // bti_idx == -1 means there was no theta trigger for this segment
      int theta_bti_group;
      int segment_number;  // position(i)
      int theta_code;
      int theta_quality;
    };

    // See documentation in DataFormats/GEMDigi/interface/GEMPadDigiCluster.h
    struct GEMData {
      GEMData() : pad(0), pad_low(0), pad_hi(0), bx(0) {}
      uint16_t pad;
      uint16_t pad_low;  // for use in clustering
      uint16_t pad_hi;   // for use in clustering
      int16_t bx;
    };

    // See documentation in DataFormats/GEMDigi/interface/ME0TriggerDigi.h
    struct ME0Data {
      ME0Data() : chamberid(0), quality(0), phiposition(0), partition(0), deltaphi(0), bend(0), bx(0) {}
      uint16_t chamberid;
      uint16_t quality;
      uint16_t phiposition;
      uint16_t partition;
      uint16_t deltaphi;
      uint16_t bend;
      uint16_t bx;
    };

    // Persistency
    TriggerPrimitive() : _id(0), _subsystem(kNSubsystems) {}

    // Constructors from DT data
    TriggerPrimitive(const DTChamberId& detid, const L1MuDTChambPhDigi& digi_phi, const int segment_number);
    TriggerPrimitive(const DTChamberId& detid, const L1MuDTChambThDigi& digi_th, const int theta_bti_group);
    TriggerPrimitive(const DTChamberId& detid,
                     const L1MuDTChambPhDigi& digi_phi,
                     const L1MuDTChambThDigi& digi_th,
                     const int theta_bti_group);

    // Constructor from CSC data
    TriggerPrimitive(const CSCDetId& detid, const CSCCorrelatedLCTDigi& digi);

    // Constructors from RPC data
    TriggerPrimitive(const RPCDetId& detid, const RPCDigi& digi);
    TriggerPrimitive(const RPCDetId& detid, const RPCRecHit& rechit);

    // Constructor from CPPF data
    TriggerPrimitive(const RPCDetId& detid, const l1t::CPPFDigi& digi);

    // Constructor from GEM data
    TriggerPrimitive(const GEMDetId& detid, const GEMPadDigiCluster& digi);

    // Constructor from ME0 data
    TriggerPrimitive(const ME0DetId& detid, const ME0TriggerDigi& digi);
    // Constructor from GE0 data
    TriggerPrimitive(const GEMDetId& detid, const ME0TriggerDigi& digi);

    // Copy constructor
    TriggerPrimitive(const TriggerPrimitive& tp);
    TriggerPrimitive& operator=(const TriggerPrimitive& tp);
    bool operator==(const TriggerPrimitive& tp) const;

    // return the subsystem we belong to
    subsystem_type subsystem() const { return _subsystem; }

    void setCMSGlobalEta(double eta) { _eta = eta; }
    void setCMSGlobalPhi(double phi) { _phi = phi; }
    void setCMSGlobalRho(double rho) { _rho = rho; }

    double getCMSGlobalEta() const { return _eta; }
    double getCMSGlobalPhi() const { return _phi; }
    double getCMSGlobalRho() const { return _rho; }

    GlobalPoint getCMSGlobalPoint() const {
      double theta = 2. * std::atan(std::exp(-_eta));
      return GlobalPoint(GlobalPoint::Cylindrical(_rho, _phi, _rho / std::tan(theta)));
    }

    // this is the relative bending angle with respect to the
    // current phi position.
    // The total angle of the track is phi + bendAngle
    void setThetaBend(double theta) { _theta = theta; }
    double getThetaBend() const { return _theta; }

    template <typename IDType>
    IDType detId() const {
      return IDType(_id);
    }

    // accessors to raw subsystem data
    void setDTData(const DTData& dt) { _dt = dt; }
    void setCSCData(const CSCData& csc) { _csc = csc; }
    void setRPCData(const RPCData& rpc) { _rpc = rpc; }
    void setGEMData(const GEMData& gem) { _gem = gem; }
    void setME0Data(const ME0Data& me0) { _me0 = me0; }

    DTData getDTData() const { return _dt; }
    CSCData getCSCData() const { return _csc; }
    RPCData getRPCData() const { return _rpc; }
    GEMData getGEMData() const { return _gem; }
    ME0Data getME0Data() const { return _me0; }

    DTData& accessDTData() { return _dt; }
    CSCData& accessCSCData() { return _csc; }
    RPCData& accessRPCData() { return _rpc; }
    GEMData& accessGEMData() { return _gem; }
    ME0Data& accessME0Data() { return _me0; }

    // consistent accessors to common information
    int getBX() const;
    int getStrip() const;
    int getWire() const;
    int getPattern() const;
    DetId rawId() const { return _id; }

    unsigned getGlobalSector() const { return _globalsector; }
    unsigned getSubSector() const { return _subsector; }

    void print(std::ostream&) const;

  private:
    // Translate to 'global' position information at the level of 60
    // degree sectors. Use CSC sectors as a template
    template <typename IDType>
    void calculateGlobalSector(const IDType& chid, unsigned& globalsector, unsigned& subsector) const {
      // Not sure if this is ever going to get implemented
      globalsector = 0;
      subsector = 0;
    }

    DTData _dt;
    CSCData _csc;
    RPCData _rpc;
    GEMData _gem;
    ME0Data _me0;

    DetId _id;

    subsystem_type _subsystem;

    unsigned _globalsector;   // [1,6] in 60 degree sectors
    unsigned _subsector;      // [1,2] in 30 degree partitions of a sector
    double _eta, _phi, _rho;  // global pseudorapidity, phi, rho
    double _theta;            // bend angle with respect to ray from (0,0,0)
  };

}  // namespace L1TMuon

#endif
