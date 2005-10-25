#ifndef DATAFORMATS_ECALUNCALIBRATEDRECHIT
#define DATAFORMATS_ECALUNCALIBRATEDRECHIT

#include <vector>
#include "DataFormats/DetId/interface/DetId.h"

using namespace cms;
class EcalUncalibratedRecHit {

  public:
   EcalUncalibratedRecHit();
   EcalUncalibratedRecHit(const DetId& detId, const double& ampl, const double& ped,
                          const double& jit, const double& chi2);

   virtual ~EcalUncalibratedRecHit();
   double amplitude() { return amplitude_; }
   double pedestal() { return pedestal_; }
   double jitter() { return jitter_; }
   double chi2() { return chi2_; }
   DetId  id() { return id_; }

 private:
   double amplitude_;   //< Reconstructed amplitude
   double pedestal_;    //< Reconstructed pedestal
   double jitter_;      //< Reconstructed time jitter
   double chi2_;        //< Chi2 of the fit
   DetId  id_;          //< Detector ID
};

typedef std::vector< EcalUncalibratedRecHit > EcalUncalibratedRecHitCollection;
#endif
