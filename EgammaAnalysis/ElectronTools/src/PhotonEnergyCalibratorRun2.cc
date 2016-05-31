#include "EgammaAnalysis/ElectronTools/interface/PhotonEnergyCalibratorRun2.h"
#include <CLHEP/Random/RandGaussQ.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

PhotonEnergyCalibratorRun2::PhotonEnergyCalibratorRun2(bool isMC, bool synchronization, 
						       std::string correctionFile
						       ) :
  isMC_(isMC), synchronization_(synchronization),
  rng_(0),
  _correctionRetriever(correctionFile) // here is opening the files and reading thecorrections
{
  if(isMC_) {
    _correctionRetriever.doScale = false; 
    _correctionRetriever.doSmearings = true;
  } else {
    _correctionRetriever.doScale = true; 
    _correctionRetriever.doSmearings = false;
  }
}

PhotonEnergyCalibratorRun2::~PhotonEnergyCalibratorRun2()
{}

void PhotonEnergyCalibratorRun2::initPrivateRng(TRandom *rnd) {
     rng_ = rnd;   
}

void PhotonEnergyCalibratorRun2::calibrate(reco::Photon &photon, unsigned int runNumber, edm::StreamID const &id) const {
  SimplePhoton simple(photon, runNumber, isMC_);
  calibrate(simple, id);
  simple.writeTo(photon);
}

void PhotonEnergyCalibratorRun2::calibrate(SimplePhoton &photon, edm::StreamID const & id) const {
    assert(isMC_ == photon.isMC());
    float smear = 0.0, scale = 1.0;
    float aeta = std::abs(photon.getEta()); //, r9 = photon.getR9();
    float et = photon.getNewEnergy()/cosh(aeta);

    scale = _correctionRetriever.ScaleCorrection(photon.getRunNumber(), photon.isEB(), photon.getR9(), aeta, et);
    smear = _correctionRetriever.getSmearingSigma(photon.getRunNumber(), photon.isEB(), photon.getR9(), aeta, et, 0., 0.); 
    
    double newEcalEnergy, newEcalEnergyError;
    if (isMC_) {
        double corr = 1.0 + smear * gauss(id);
        newEcalEnergy      = photon.getNewEnergy() * corr;
        newEcalEnergyError = std::hypot(photon.getNewEnergyError() * corr, smear * newEcalEnergy);
    } else {
        newEcalEnergy      = photon.getNewEnergy() * scale;
        newEcalEnergyError = std::hypot(photon.getNewEnergyError() * scale, smear * newEcalEnergy);
    }
    photon.setNewEnergy(newEcalEnergy); 
    photon.setNewEnergyError(newEcalEnergyError); 

}

double PhotonEnergyCalibratorRun2::gauss(edm::StreamID const& id) const {
  if (synchronization_) return 1.0;
  if (rng_) {
    return rng_->Gaus();
  } else {
    edm::Service<edm::RandomNumberGenerator> rng;
    if ( !rng.isAvailable() ) {
      throw cms::Exception("Configuration")
	<< "XXXXXXX requires the RandomNumberGeneratorService\n"
	"which is not present in the configuration file.  You must add the service\n"
	"in the configuration file or remove the modules that require it.";
    }
    CLHEP::RandGaussQ gaussDistribution(rng->getEngine(id), 0.0, 1.0);
    return gaussDistribution.fire();
  }
}

