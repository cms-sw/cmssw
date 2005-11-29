#ifndef METReco_MET_h
#define METReco_MET_h
#include <Rtypes.h>
#include <cmath>
namespace reco {

  class MET {
  public:
    MET() { }
    MET( double mex, double mey );
    double mEx() const { return mEx_; }
    double mEy() const { return mEy_; }
    double mEt() const { return sqrt( mEx_ * mEx_ + mEy_ * mEy_ ); }
    double phi() const { return atan2( mEy_, mEx_ ); }
    double uncorrectedMEx() const { return uncorrectedMEx_; }
    double uncorrectedMEy() const { return uncorrectedMEy_; }
  private:
    Double32_t mEx_, mEy_;
    Double32_t uncorrectedMEx_, uncorrectedMEy_;
  };

}

#endif
