#ifndef DATAFORMATS_FTLUNCALIBRATEDRECHIT
#define DATAFORMATS_FTLUNCALIBRATEDRECHIT

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

class FTLUncalibratedRecHit {

  public:
  
  typedef DetId key_type;

  enum Flags {
          kGood=-1,                 // channel is good (mutually exclusive with other states)  setFlagBit(kGood) reset flags_ to zero 
          kPoorReco,                // channel has been badly reconstructed (e.g. bad shape, bad chi2 etc.)
          kSaturated,               // saturated channel
          kOutOfTime               // channel out of time
  };

  FTLUncalibratedRecHit();
  FTLUncalibratedRecHit(const DetId& detId, float ampl, float time, uint32_t flags = 0, uint32_t aux = 0);

  ~FTLUncalibratedRecHit();
  float amplitude() const { return amplitude_; }
  float time() const { return time_; }
  

  uint32_t flags() const { return flags_; }
  float timeError() const;
  uint8_t timeErrorBits() const;
  DetId  id() const { return id_; }

  void setAmplitude( float amplitude ) { amplitude_ = amplitude; }
  void setTime( float time ) { time_ = time; }

  void setTimeError( float timeErr );
  void setFlags( uint32_t flags ) { flags_ = flags; }
  void setId( DetId id ) { id_ = id; }
  void setAux( uint32_t aux ) { aux_ = aux; }
  void setFlagBit(Flags flag);
  bool checkFlag(Flags flag) const;

  bool isSaturated() const;
  bool isTimeValid() const;
  bool isTimeErrorValid() const;

 private:
  float amplitude_;    //< Reconstructed amplitude
  float time_;       //< Reconstructed time jitter
  uint32_t flags_;     //< flag to be propagated to RecHit
  uint32_t aux_;       //< aux word; first 8 bits contain time (jitter) error
  DetId  id_;          //< Detector ID
};

#endif
