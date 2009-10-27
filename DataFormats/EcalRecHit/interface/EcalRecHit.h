#ifndef DATAFORMATS_ECALRECHIT_H
#define DATAFORMATS_ECALRECHIT_H 1

#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"

/** \class EcalRecHit
 *  
 * $id: $
 * \author P. Meridiani INFN Roma1
 */

class EcalRecHit : public CaloRecHit {
public:
  typedef DetId key_type;

  // recHit flags
  enum Flags { 
          kGood,                     // channel ok, the energy and time measurement are reliable
          kPoorReco,                 // the energy is available from the UncalibRecHit, but approximate (bad shape, large chi2)
          kOutOfTime,                // the energy is available from the UncalibRecHit (sync reco), but the event is out of time
          kFaultyHardware,           // The energy is available from the UncalibRecHit, channel is faulty at some hardware level (e.g. noisy)
          kPoorCalib,                // the energy is available from the UncalibRecHit, but the calibration of the channel is poor
          kSaturated,                // saturated channel (recovery not tried)
          kLeadingEdgeRecovered,     // saturated channel: energy estimated from the leading edge before saturation
          kNeighboursRecovered,      // saturated/isolated dead: energy estimated from neighbours
          kTowerRecovered,           // channel in TT with no data link, info retrieved from Trigger Primitive
          kDead                      // channel is dead and any recovery fails
  };

  EcalRecHit();
  EcalRecHit(const DetId& id, float energy, float time, uint32_t flags = 0);
  /// get the id
  // For the moment not returning a specific id for subdetector
  DetId id() const { return DetId(detid());}
  bool isRecovered() const;
  uint32_t recoFlag() const { return 0xF & flags(); }
  float chi2Prob() const;
  // set the energy for out of time events
  // (only energy >= 0 will be stored)
  float outOfTimeEnergy() const;
  void setRecoFlag( uint32_t flag );
  void setChi2Prob( float chi2Prob );
  void setOutOfTimeEnergy( float energy );
};

std::ostream& operator<<(std::ostream& s, const EcalRecHit& hit);

#endif
