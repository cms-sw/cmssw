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
  correctionRetriever_(correctionFile) // here is opening the files and reading thecorrections
{
  if(isMC_) {
    correctionRetriever_.doScale = false;
    correctionRetriever_.doSmearings = true;
  } else {
    correctionRetriever_.doScale = true;
    correctionRetriever_.doSmearings = false;
  }
}

PhotonEnergyCalibratorRun2::~PhotonEnergyCalibratorRun2()
{}

void PhotonEnergyCalibratorRun2::initPrivateRng(TRandom *rnd)
{
  rng_ = rnd;
}

std::vector<float> PhotonEnergyCalibratorRun2::calibrate(reco::Photon &photon, unsigned int runNumber, 
							 const EcalRecHitCollection *recHits, edm::StreamID const & id, int eventIsMC) const
{
  float smear = 0.0, scale = 1.0;
  float aeta = std::abs(photon.superCluster()->eta());
  float et = photon.getCorrectedEnergy(reco::Photon::P4type::regression2) / cosh(aeta);

  if (et < 1.0) {
    return std::vector<float>(10, photon.getCorrectedEnergy(reco::Photon::P4type::regression2));
  }

  DetId seedDetId = photon.superCluster()->seed()->seed();
  EcalRecHitCollection::const_iterator seedRecHit = recHits->find(seedDetId);
  unsigned int gainSeedSC = 12;
  if (seedRecHit != recHits->end()) {
    if(seedRecHit->checkFlag(EcalRecHit::kHasSwitchToGain6)) gainSeedSC = 6;
    if(seedRecHit->checkFlag(EcalRecHit::kHasSwitchToGain1)) gainSeedSC = 1;
  }
  scale = correctionRetriever_.ScaleCorrection(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC);
  smear = correctionRetriever_.getSmearingSigma(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 0., 0.);

  // This always to be carefully thought
  float scale_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 7);
  float scale_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 7);
  float resol_up  = correctionRetriever_.getSmearingSigma(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 1., 0.);
  float resol_dn  = correctionRetriever_.getSmearingSigma(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, -1., 0.);
  
  float scale_stat_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 1);
  float scale_stat_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 1);
  float scale_syst_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 2);
  float scale_syst_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 2);
  float scale_gain_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 4);
  float scale_gain_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 4);
  float resol_rho_up  = correctionRetriever_.getSmearingSigma(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 1., 0.);
  float resol_rho_dn  = correctionRetriever_.getSmearingSigma(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, -1., 0.);
  float resol_phi_up  = correctionRetriever_.getSmearingSigma(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 0., 1.);
  float resol_phi_dn  = correctionRetriever_.getSmearingSigma(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 0., -1.);
  std::vector<float> uncertainties;

  double newEcalEnergy = photon.getCorrectedEnergy(reco::Photon::P4type::regression2);
  double newEcalEnergyError = photon.getCorrectedEnergyError(reco::Photon::P4type::regression2);

  if ((eventIsMC < 0 && isMC_) || (eventIsMC == 1)) {
    double rndm = gauss(id); 
    double corr = 1.0 + smear * rndm;
    double corr_rho_up = 1.0 + resol_rho_up * rndm;
    double corr_rho_dn = 1.0 + resol_rho_dn * rndm;
    double corr_phi_up = 1.0 + resol_phi_up * rndm;
    double corr_phi_dn = 1.0 + resol_phi_dn * rndm;

    double corr_up = 1.0 + resol_up * rndm;
    double corr_dn = 1.0 + resol_dn * rndm;

    newEcalEnergy      = newEcalEnergy * corr;
    newEcalEnergyError = std::hypot(newEcalEnergyError * corr, smear * newEcalEnergy);

    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy * corr_rho_up/corr);
    uncertainties.push_back(newEcalEnergy * corr_rho_dn/corr);
    uncertainties.push_back(newEcalEnergy * corr_phi_up/corr);
    uncertainties.push_back(newEcalEnergy * corr_phi_dn/corr);
    
    // The total variation
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy * corr_up/corr);
    uncertainties.push_back(newEcalEnergy * corr_dn/corr);

  } else if ((eventIsMC < 0 && !isMC_) || (eventIsMC == 0)) {

    newEcalEnergy      = newEcalEnergy * scale;
    newEcalEnergyError = std::hypot(newEcalEnergyError * scale, smear * newEcalEnergy);

    uncertainties.push_back(newEcalEnergy * scale_stat_up/scale);
    uncertainties.push_back(newEcalEnergy * scale_stat_dn/scale);
    uncertainties.push_back(newEcalEnergy * scale_syst_up/scale);
    uncertainties.push_back(newEcalEnergy * scale_syst_dn/scale);
    uncertainties.push_back(newEcalEnergy * scale_gain_up/scale);
    uncertainties.push_back(newEcalEnergy * scale_gain_dn/scale);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);

    // The total variation
    uncertainties.push_back(newEcalEnergy * scale_up/scale);
    uncertainties.push_back(newEcalEnergy * scale_dn/scale);
    uncertainties.push_back(newEcalEnergy);
    uncertainties.push_back(newEcalEnergy);

  }
  photon.setCorrectedEnergy(reco::Photon::P4type::regression2, newEcalEnergy, newEcalEnergyError, true);
  
  return uncertainties;
  
}

double PhotonEnergyCalibratorRun2::gauss(edm::StreamID const& id) const
{
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

