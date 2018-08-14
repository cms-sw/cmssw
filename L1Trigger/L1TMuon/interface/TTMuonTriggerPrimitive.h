#ifndef __L1TMuon_TTMuonTriggerPrimitive_h__
#define __L1TMuon_TTMuonTriggerPrimitive_h__

//
// This class implements a layer for Phase 2 Tracker trigger primitive
// analogous to the class MuonTriggerPrimitive.
//

#include <cstdint>
#include <vector>
#include <iosfwd>

// DetId
#include "DataFormats/DetId/interface/DetId.h"
// Global point
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
// Track trigger data formats
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


namespace L1TMuon {

  class TTTriggerPrimitive {
  public:
    // Define the subsystems that we have available (none at the moment).
    // Start at 20 for TTMuonTriggerPrimitive to avoid collision with MuonTriggerPrimitive
    enum subsystem_type{kTT = 20, kNSubsystems};

    // Rename these
    typedef TTStub<Ref_Phase2TrackerDigi_>  TTDigi;
    typedef DetId                           TTDetId;

    // Define the raw data
    struct TTData {
      TTData() : row_f(0.), col_f(0.), bend(0), bx(0) {}
      float row_f;  // why float?
      float col_f;  // why float?
      int bend;
      int16_t bx;
    };

    // Persistency
    TTTriggerPrimitive(): _subsystem(kNSubsystems) {}

    // Constructor from track trigger digi
    TTTriggerPrimitive(const TTDetId& detid, const TTDigi& digi);

    // Copy constructor
    TTTriggerPrimitive(const TTTriggerPrimitive&);

    // Destructor
    ~TTTriggerPrimitive() {}

    // Assignment operator
    TTTriggerPrimitive& operator=(const TTTriggerPrimitive& tp);

    // Equality operator
    bool operator==(const TTTriggerPrimitive& tp) const;

    // Subsystem type
    const subsystem_type subsystem() const { return _subsystem; }

    // Global coordinates
    const double getCMSGlobalEta() const { return _eta; }
    void         setCMSGlobalEta(const double eta) { _eta = eta; }
    const double getCMSGlobalPhi() const { return _phi; }
    void         setCMSGlobalPhi(const double phi) { _phi = phi; }
    const double getCMSGlobalRho() const { return _rho; }
    void         setCMSGlobalRho(const double rho) { _rho = rho; }

    const GlobalPoint getCMSGlobalPoint() const {
      double theta = 2. * atan( exp(-_eta) );
      return GlobalPoint( GlobalPoint::Cylindrical( _rho, _phi, _rho/tan(theta)) );
    };


    // Detector id
    TTDetId detId() const { return _id; }

    TTDetId rawId() const { return detId(); }

    // Accessors to raw data
    void setTTData(const TTData& data) { _data = data; }

    const TTData  getTTData()  const { return _data; }

    TTData&  accessTTData()  { return _data; }

    // Accessors to common information
    const int getStrip() const;
    const int getSegment() const;
    const int getBend() const;
    const int getBX() const;

    const unsigned getGlobalSector() const { return _globalsector; }
    const unsigned getSubSector() const { return _subsector; }

    void print(std::ostream&) const;

  private:
    // Translate to 'global' position information at the level of 60
    // degree sectors. Use CSC sectors as a template
    void calculateTTGlobalSector(const TTDetId& detid,
                                 unsigned& globalsector,
                                 unsigned& subsector );

    TTData _data;

    TTDetId _id;

    subsystem_type _subsystem;

    unsigned _globalsector; // [1,6] in 60 degree sectors
    unsigned _subsector; // [1,2] in 30 degree partitions of a sector
    double _eta, _phi, _rho;  // global pseudorapidity, phi, rho
    double _theta; // bend angle with respect to ray from (0,0,0). // NOT USED
  };

}  // namespace L1TMuon

#endif
