#ifndef DATAFORMATS_ECALUNCALIBRATEDRECHIT
#define DATAFORMATS_ECALUNCALIBRATEDRECHIT

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

class EcalUncalibratedRecHit {

  public:
  
  typedef DetId key_type;

  enum Flags {
          kGood,                 // channel is good
          kPoorReco,             // channel has been badly reconstructed (e.g. bad shape, bad chi2 etc.)
          kSaturated,            // saturated channel
          kOutOfTime,            // channel out of time
          kLeadingEdgeRecovered, // saturated channel: energy estimated from the leading edge before saturation
          kFake                  // the signal in the channel is a fake (e.g. a so-called spike)
  };

  EcalUncalibratedRecHit();
  EcalUncalibratedRecHit(const DetId& detId, double ampl, double ped,
                          double jit, double chi2, uint32_t flags = 0, uint32_t aux = 0);

  virtual ~EcalUncalibratedRecHit();
  double amplitude() const { return amplitude_; }
  double pedestal() const { return pedestal_; }
  double jitter() const { return jitter_; }
  double chi2() const { return chi2_; }
  uint32_t recoFlag() const { return 0xF & flags_; }
  float  outOfTimeEnergy() const;
  float  outOfTimeChi2() const;
  DetId  id() const { return id_; }

  void setAmplitude( double amplitude ) { amplitude_ = amplitude; }
  void setPedestal( double pedestal ) { pedestal_ = pedestal; }
  void setJitter( double jitter ) { jitter_ = jitter; }
  void setChi2( double chi2 ) { chi2_ = chi2; }
  void setFlags( uint32_t flags ) { flags_ = flags; }
  void setRecoFlag( uint32_t flags );
  // set the energy for out of time events
  // (only energy >= 0 will be stored)
  void setOutOfTimeEnergy( float energy );
  void setOutOfTimeChi2( float chi2 );
  void setId( DetId id ) { id_ = id; }
  
  bool isSaturated() const;

 private:
  double amplitude_;   //< Reconstructed amplitude
  double pedestal_;    //< Reconstructed pedestal
  double jitter_;      //< Reconstructed time jitter
  double chi2_;        //< Chi2 of the fit
  uint32_t flags_;     //< flag to be propagated to RecHit
  uint32_t aux_;
  DetId  id_;          //< Detector ID
};

#endif
