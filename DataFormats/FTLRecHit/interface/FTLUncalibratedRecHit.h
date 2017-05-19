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
  FTLUncalibratedRecHit(const DetId& detId, float ampl, float time, float timeError, unsigned char flags = 0);

  ~FTLUncalibratedRecHit();
  float amplitude() const { return amplitude_; }
  float time() const { return time_; }

  float timeError() const {return timeError_; }

  DetId  id() const { return id_; }

  void setAmplitude( float amplitude ) { amplitude_ = amplitude; }
  void setTime( float time ) { time_ = time; }

  void setTimeError( float timeErr ) { timeError_ = timeErr; }
  void setId( DetId id ) { id_ = id; }
  void setFlagBit(Flags flag);
  bool checkFlag(Flags flag) const;

  bool isTimeValid() const;
  bool isTimeErrorValid() const;

  bool isSaturated() const;
  
 private:
  float amplitude_;    //< Reconstructed amplitude
  float time_;       //< Reconstructed time jitter
  float timeError_;  
  DetId  id_;          //< Detector ID
  unsigned char flags_;
};

#endif
