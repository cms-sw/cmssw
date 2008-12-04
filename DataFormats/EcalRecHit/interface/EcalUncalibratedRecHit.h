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
                          const double& jit, const double& chi2);

  virtual ~EcalUncalibratedRecHit();
  double amplitude() const { return amplitude_; }
  double pedestal() const { return pedestal_; }
  double jitter() const { return jitter_; }
  double chi2() const { return chi2_; }
  DetId  id() const { return id_; }
  bool isSaturated() const;

 private:
  double amplitude_;   //< Reconstructed amplitude
  double pedestal_;    //< Reconstructed pedestal
  double jitter_;      //< Reconstructed time jitter
  double chi2_;        //< Chi2 of the fit
  DetId  id_;          //< Detector ID
};

#endif
