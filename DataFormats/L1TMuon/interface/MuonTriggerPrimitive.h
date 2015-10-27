#ifndef __L1TMUON_TRIGGERPRIMITIVE_H__
#define __L1TMUON_TRIGGERPRIMITIVE_H__
// 
// Class: l1t::MuonTriggerPrimitive
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

#include <boost/cstdint.hpp>
#include <vector>
#include <iostream>

//DetId
#include "DataFormats/DetId/interface/DetId.h"

// DT digi types
class DTChamberId;
class L1MuDTChambPhDigi;
class L1MuDTChambThDigi;

// CSC digi types
class CSCCorrelatedLCTDigi;
class CSCDetId;

// RPC digi types
class RPCDigiL1Link;
class RPCDetId;


#include <map>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace l1t {

  class MuonTriggerPrimitive;

  typedef std::vector<MuonTriggerPrimitive> MuonTriggerPrimitiveCollection;  
  typedef edm::Ref<MuonTriggerPrimitiveCollection> MuonTriggerPrimitiveRef;
  typedef std::vector<MuonTriggerPrimitiveRef> MuonTriggerPrimitiveList;
  typedef edm::Ptr<MuonTriggerPrimitive> MuonTriggerPrimitivePtr;
  typedef std::map<unsigned,MuonTriggerPrimitiveList> MuonTriggerPrimitiveStationMap;





  class MuonTriggerPrimitive {
  public:
    // define the subsystems that we have available
    enum subsystem_type{kDT,kCSC,kRPC,kNSubsystems};
    
    // define the data we save locally from each subsystem type
    // variables in these structs keep their colloquial meaning
    // within a subsystem
    // for RPCs you have to unroll the digi-link and raw det-id
    struct RPCData {
      RPCData() : strip(0), layer(0), bx(0) {}
      unsigned strip;
      unsigned layer;
      uint16_t bx;
    };

    struct CSCData {
      CSCData() : trknmb(0), valid(0), quality(0), keywire(0), strip(0),
		  pattern(0), bend(0), bx(0), mpclink(0), bx0(0), syncErr(0),
		  cscID(0) {}
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
    };

    struct DTData {
      DTData() : bx(0), wheel(0), sector(0), station(0), radialAngle(0),
		 bendingAngle(0), qualityCode(0), Ts2TagCode(0), BxCntCode(0),
		 theta_bti_group(0), segment_number(0), theta_code(0),
		 theta_quality(0) {}
      // from ChambPhDigi (corresponds to a TRACO)
      // this gives us directly the phi
      int bx; // relative? bx number
      int wheel; // wheel number -3,-2,-1,1,2,3
      int sector; // 1-12 in DT speak (these correspond to CSC sub-sectors)
      int station; // 1-4 radially outwards
      int radialAngle; // packed phi in a sector
      int bendingAngle; // angle of segment relative to chamber
      int qualityCode; // need to decode
      int Ts2TagCode; // ??
      int BxCntCode; // ????
      // from ChambThDigi (corresponds to a BTI)
      // we have to root out the eta manually
      // theta super layer == SL 1
      // station four has no theta super-layer
      // bti_idx == -1 means there was no theta trigger for this segment
      int theta_bti_group;
      int segment_number; // position(i)
      int theta_code;
      int theta_quality;
    };
    
    //Persistency
    MuonTriggerPrimitive(): _subsystem(kNSubsystems) {}
      
    //DT      
    MuonTriggerPrimitive(const DTChamberId&,		     
		     const L1MuDTChambPhDigi&,
		     const int segment_number);
    MuonTriggerPrimitive(const DTChamberId&,		     
		     const L1MuDTChambThDigi&,
		     const int segment_number);
    MuonTriggerPrimitive(const DTChamberId&,		     
		     const L1MuDTChambPhDigi&,
		     const L1MuDTChambThDigi&,
		     const int theta_bti_group);    
    //CSC
    MuonTriggerPrimitive(const CSCDetId&,
		     const CSCCorrelatedLCTDigi&);
    //RPC
    MuonTriggerPrimitive(const RPCDetId& detid,
		     const unsigned strip,
		     const unsigned layer,
		     const uint16_t bx);
    
    //copy
    MuonTriggerPrimitive(const MuonTriggerPrimitive&);

    MuonTriggerPrimitive& operator=(const MuonTriggerPrimitive& tp);
    bool operator==(const MuonTriggerPrimitive& tp) const;

    // return the subsystem we belong to
    const subsystem_type subsystem() const { return _subsystem; }    

    const double getCMSGlobalEta() const { return _eta; }
    void   setCMSGlobalEta(const double eta) { _eta = eta; }
    const double getCMSGlobalPhi() const { return _phi; }    
    void   setCMSGlobalPhi(const double phi) { _phi = phi; }

    // this is the relative bending angle with respect to the 
    // current phi position. 
    // The total angle of the track is phi + bendAngle
    void setThetaBend(const double theta) { _theta = theta; }
    double getThetaBend() const { return _theta; }

    template<typename IDType>
      IDType detId() const { return IDType(_id); }

    // accessors to raw subsystem data
    const DTData  getDTData()  const { return _dt;  }
    const CSCData getCSCData() const { return _csc; }
    const RPCData getRPCData() const { return _rpc; }      
    
    // consistent accessors to common information    
    const int getBX() const;
    const int getStrip() const;
    const int getWire() const;
    const int getPattern() const;
    const DetId rawId() const {return _id;};
    const int Id() const;
    
    const unsigned getGlobalSector() const { return _globalsector; } 
    const unsigned getSubSector() const { return _subsector; } 
    
    void print(std::ostream&) const;
    
  private:
    // Translate to 'global' position information at the level of 60
    // degree sectors. Use CSC sectors as a template
    void calculateDTGlobalSector(const DTChamberId& chid, 
				 unsigned& global_sector, 
				 unsigned& subsector );
    void calculateCSCGlobalSector(const CSCDetId& chid, 
				  unsigned& global_sector, 
				  unsigned& subsector );
    void calculateRPCGlobalSector(const RPCDetId& chid, 
				  unsigned& global_sector, 
				  unsigned& subsector );
      
    DTData  _dt;
    CSCData _csc;
    RPCData _rpc;
    
    DetId _id;
    
    subsystem_type _subsystem;

    unsigned _globalsector; // [1,6] in 60 degree sectors
    unsigned _subsector; // [1,2] in 30 degree partitions of a sector 
    double _eta,_phi; // global pseudorapidity, phi
    double _theta; // bend angle with respect to ray from (0,0,0)    
  };

}

#endif

