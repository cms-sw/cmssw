#ifndef DATAFORMATS_ECALUNCALIBRATEDRECHIT
#define DATAFORMATS_ECALUNCALIBRATEDRECHIT

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

class EcalUncalibratedRecHit {

  public:
  
  typedef DetId key_type;

  static const double kSATURATED;
  static const float kPRECISION;

  EcalUncalibratedRecHit();
  EcalUncalibratedRecHit(const DetId& detId, const double& ampl, const double& ped,
                          const double& jit, const double& chi2, const uint32_t &flags = 0);

  virtual ~EcalUncalibratedRecHit();
  double amplitude() const { return amplitude_; }
  double pedestal() const { return pedestal_; }
  double jitter() const { return jitter_; }
  double chi2() const { return chi2_; }
  uint32_t flags() const { return flags_; }
  DetId  id() const { return id_; }

  void setAmplitude( double amplitude ) { amplitude_ = amplitude; }
  void setPedestal( double pedestal ) { pedestal_ = pedestal; }
  void setJitter( double jitter ) { jitter_ = jitter; }
  void setChi2( double chi2 ) { chi2_ = chi2; }
  void setFlags( uint32_t flags ) { flags_ = flags; }
  void setId( DetId id ) { id_ = id; }
  
  bool isSaturated() const;

 private:
  double amplitude_;   //< Reconstructed amplitude
  double pedestal_;    //< Reconstructed pedestal
  double jitter_;      //< Reconstructed time jitter
  double chi2_;        //< Chi2 of the fit
  uint32_t flags_;     //< flag to be propagated to RecHit
  DetId  id_;          //< Detector ID
};

#endif
