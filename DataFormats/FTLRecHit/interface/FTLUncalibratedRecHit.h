#ifndef DATAFORMATS_FTLUNCALIBRATEDRECHIT
#define DATAFORMATS_FTLUNCALIBRATEDRECHIT

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

class FTLUncalibratedRecHit {

  public:
  
  typedef DetId key_type;

  enum Flags {
          kGood=-1,   // channel is good (mutually exclusive with other states)  setFlagBit(kGood) reset flags_ to zero
          kPoorReco,  // channel has been badly reconstructed (e.g. bad shape, bad chi2 etc.)
          kSaturated, // saturated channel
          kOutOfTime  // channel out of time
  };

  FTLUncalibratedRecHit();
  FTLUncalibratedRecHit(const DetId& detId, std::pair <float,float> ampl,
			std::pair <float,float> time, float timeError, unsigned char flags = 0);
  FTLUncalibratedRecHit(const DetId& detId, uint8_t row, uint8_t column, 
			std::pair <float,float> ampl, std::pair <float,float> time,
			float timeError, unsigned char flags = 0);

  ~FTLUncalibratedRecHit();
  std::pair <float,float> amplitude() const { return amplitude_; }
  std::pair <float,float> time() const { return time_; }

  float timeError() const {return timeError_; }

  unsigned char flags() const { return flags_; };

  DetId  id() const { return id_; }
  int row() const { return row_; }
  int column() const { return column_; }

  void setAmplitude( std::pair <float,float> amplitude ) { amplitude_ = amplitude; }
  void setTime( std::pair <float,float> time ) { time_ = time; }

  void setTimeError( float timeErr ) { timeError_ = timeErr; }
  void setId( DetId id ) { id_ = id; }
  void setFlagBit(Flags flag);
  bool checkFlag(Flags flag) const;

  bool isTimeValid() const;
  bool isTimeErrorValid() const;

  bool isSaturated() const;
  
 private:
  std::pair <float,float> amplitude_;
  std::pair <float,float> time_;
  float timeError_;  
  DetId  id_;
  uint8_t row_, column_;
  unsigned char flags_;
};

#endif
