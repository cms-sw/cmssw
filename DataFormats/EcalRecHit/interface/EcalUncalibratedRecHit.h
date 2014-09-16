#ifndef DATAFORMATS_ECALUNCALIBRATEDRECHIT
#define DATAFORMATS_ECALUNCALIBRATEDRECHIT

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

class EcalUncalibratedRecHit {

  public:
  
  typedef DetId key_type;

  enum Flags {
          kGood=-1,                 // channel is good (mutually exclusive with other states)  setFlagBit(kGood) reset flags_ to zero 
          kPoorReco,                // channel has been badly reconstructed (e.g. bad shape, bad chi2 etc.)
          kSaturated,               // saturated channel
          kOutOfTime,               // channel out of time
          kLeadingEdgeRecovered,    // saturated channel: energy estimated from the leading edge before saturation
          kHasSwitchToGain6,        // at least one data frame is in G6
          kHasSwitchToGain1         // at least one data frame is in G1
          
  };

  EcalUncalibratedRecHit() :
      amplitude_(0.), pedestal_(0.), jitter_(0.), chi2_(10000.), OOTamplitude_(0.), OOTchi2_(10000.), flags_(0), aux_(0) { }

  EcalUncalibratedRecHit(const DetId& id, float ampl, float ped,
                          float jit, float chi2, uint32_t flags = 0, uint32_t aux = 0):
     amplitude_(ampl), pedestal_(ped), jitter_(jit), chi2_(chi2), OOTamplitude_(0.), OOTchi2_(10000.), flags_(flags), aux_(aux), id_(id) { }


  float amplitude() const { return amplitude_; }
  float pedestal() const { return pedestal_; }
  float jitter() const { return jitter_; }
  float chi2() const { return chi2_; }
  float outOfTimeEnergy() const { return OOTamplitude_; }
  float outOfTimeChi2() const { return OOTchi2_; }

  uint32_t flags() const { return flags_; }
  float jitterError() const;
  uint8_t jitterErrorBits() const;
  DetId  id() const { return id_; }

  void setAmplitude( float amplitude ) { amplitude_ = amplitude; }
  void setPedestal( float pedestal ) { pedestal_ = pedestal; }
  void setJitter( float jitter ) { jitter_ = jitter; }
  void setChi2( float chi2 ) { chi2_ = chi2; }
  void setOutOfTimeEnergy( float energy ) { OOTamplitude_ = energy; }
  void setOutOfTimeChi2( float chi2 ){ OOTchi2_ = chi2; }

  void setJitterError( float jitterErr );
  void setFlags( uint32_t flags ) { flags_ = flags; }
  void setId( DetId id ) { id_ = id; }
  void setAux( uint32_t aux ) { aux_ = aux; }
  void setFlagBit(Flags flag);
  bool checkFlag(Flags flag) const;

  bool isSaturated() const;
  bool isJitterValid() const;
  bool isJitterErrorValid() const;

 private:
  float amplitude_;    //< Reconstructed amplitude
  float pedestal_;     //< Reconstructed pedestal
  float jitter_;       //< Reconstructed time jitter
  float chi2_;         //< Chi2 of the pulse
  float OOTamplitude_; //< Out-Of-Time reconstructed amplitude 
  float OOTchi2_;      //< Out-Of-Time Chi2 
  uint32_t flags_;     //< flag to be propagated to RecHit
  uint32_t aux_;       //< aux word; first 8 bits contain time (jitter) error
  DetId  id_;          //< Detector ID
};

#endif
