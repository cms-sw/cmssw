#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronMomentumCorrector.h"

#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"

/****************************************************************************
 *
 * Class based E-p combination for the final electron momentum. It relies on
 * the electron classification and on the dtermination of track momentum and ecal
 * supercluster energy errors. The track momentum error is taken from the gsf fit.
 * The ecal supercluster energy error is taken from a class dependant parametrisation
 * of the energy resolution.
 *
 *
 * \author Federico Ferri - INFN Milano, Bicocca university
 * \author Ivica Puljak - FESB, Split
 * \author Stephanie Baffioni - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 *
 *
 ****************************************************************************/

egamma::ElectronMomentum egamma::correctElectronMomentum(reco::GsfElectron const& electron,
                                                         TrajectoryStateOnSurface const& vtxTsos) {
  int elClass = electron.classification();

  //=======================================================================================
  // cluster energy
  //=======================================================================================

  float scEnergy = electron.correctedEcalEnergy();
  float errorEnergy = electron.correctedEcalEnergyError();

  //=======================================================================================
  // track  momentum
  //=======================================================================================

  // basic values
  float trackMomentum = electron.trackMomentumAtVtx().R();
  //float errorTrackMomentum = 999. ;

  // tracker momentum scale corrections (Mykhailo Dalchenko)
  double scale = 1.;
  if (electron.isEB()) {
    if (elClass == 0) {
      scale = 1. / (0.00104 * sqrt(trackMomentum) + 1);
    }
    if (elClass == 1) {
      scale = 1. / (0.0017 * sqrt(trackMomentum) + 0.9986);
    }
    if (elClass == 3) {
      scale = 1. / (1.004 - 0.00021 * trackMomentum);
    }
    if (elClass == 4) {
      scale = 0.995;
    }
  } else if (electron.isEE()) {
    if (elClass == 3) {
      scale = 1. / (1.01432 - 0.00201872 * trackMomentum + 0.0000142621 * trackMomentum * trackMomentum);
    }
    if (elClass == 4) {
      scale = 1. / (0.996859 - 0.000345347 * trackMomentum);
    }
  }
  if (scale < 0.)
    scale = 1.;  // CC added protection
  trackMomentum = trackMomentum * scale;

  // error (must be done after trackMomentum rescaling)
  MultiGaussianState1D qpState(MultiGaussianStateTransform::multiState1D(vtxTsos, 0));
  GaussianSumUtilities1D qpUtils(qpState);
  float errorTrackMomentum = trackMomentum * trackMomentum * sqrt(qpUtils.mode().variance());

  //=======================================================================================
  // combination
  //=======================================================================================

  float finalMomentum = electron.p4().t();  // initial
  float finalMomentumError = 999.;

  // first check for large errors

  if (errorTrackMomentum / trackMomentum > 0.5 && errorEnergy / scEnergy <= 0.5) {
    finalMomentum = scEnergy;
    finalMomentumError = errorEnergy;
  } else if (errorTrackMomentum / trackMomentum <= 0.5 && errorEnergy / scEnergy > 0.5) {
    finalMomentum = trackMomentum;
    finalMomentumError = errorTrackMomentum;
  } else if (errorTrackMomentum / trackMomentum > 0.5 && errorEnergy / scEnergy > 0.5) {
    if (errorTrackMomentum / trackMomentum < errorEnergy / scEnergy) {
      finalMomentum = trackMomentum;
      finalMomentumError = errorTrackMomentum;
    } else {
      finalMomentum = scEnergy;
      finalMomentumError = errorEnergy;
    }
  }

  // then apply the combination algorithm
  else {
    // calculate E/p and corresponding error
    float eOverP = scEnergy / trackMomentum;
    float errorEOverP = sqrt((errorEnergy / trackMomentum) * (errorEnergy / trackMomentum) +
                             (scEnergy * errorTrackMomentum / trackMomentum / trackMomentum) *
                                 (scEnergy * errorTrackMomentum / trackMomentum / trackMomentum));

    bool eleIsNotInCombination = false;
    if ((eOverP > 1 + 2.5 * errorEOverP) || (eOverP < 1 - 2.5 * errorEOverP) || (eOverP < 0.8) || (eOverP > 1.3)) {
      eleIsNotInCombination = true;
    }
    if (eleIsNotInCombination) {
      if (eOverP > 1) {
        finalMomentum = scEnergy;
        finalMomentumError = errorEnergy;
      } else {
        if (elClass == reco::GsfElectron::GOLDEN) {
          finalMomentum = scEnergy;
          finalMomentumError = errorEnergy;
        }
        if (elClass == reco::GsfElectron::BIGBREM) {
          if (scEnergy < 36) {
            finalMomentum = trackMomentum;
            finalMomentumError = errorTrackMomentum;
          } else {
            finalMomentum = scEnergy;
            finalMomentumError = errorEnergy;
          }
        }
        if (elClass == reco::GsfElectron::BADTRACK) {
          finalMomentum = scEnergy;
          finalMomentumError = errorEnergy;
        }
        if (elClass == reco::GsfElectron::SHOWERING) {
          if (scEnergy < 30) {
            finalMomentum = trackMomentum;
            finalMomentumError = errorTrackMomentum;
          } else {
            finalMomentum = scEnergy;
            finalMomentumError = errorEnergy;
          }
        }
        if (elClass == reco::GsfElectron::GAP) {
          if (scEnergy < 60) {
            finalMomentum = trackMomentum;
            finalMomentumError = errorTrackMomentum;
          } else {
            finalMomentum = scEnergy;
            finalMomentumError = errorEnergy;
          }
        }
      }
    }

    else {
      // combination
      finalMomentum = (scEnergy / errorEnergy / errorEnergy + trackMomentum / errorTrackMomentum / errorTrackMomentum) /
                      (1 / errorEnergy / errorEnergy + 1 / errorTrackMomentum / errorTrackMomentum);
      float finalMomentumVariance = 1 / (1 / errorEnergy / errorEnergy + 1 / errorTrackMomentum / errorTrackMomentum);
      finalMomentumError = sqrt(finalMomentumVariance);
    }
  }

  //=======================================================================================
  // final set
  //=======================================================================================

  auto const& oldMomentum = electron.p4();

  return {{oldMomentum.x() * finalMomentum / oldMomentum.t(),
           oldMomentum.y() * finalMomentum / oldMomentum.t(),
           oldMomentum.z() * finalMomentum / oldMomentum.t(),
           finalMomentum},
          errorTrackMomentum,
          finalMomentumError};
}
