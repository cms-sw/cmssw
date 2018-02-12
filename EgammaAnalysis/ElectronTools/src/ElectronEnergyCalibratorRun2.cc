#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibratorRun2.h"
#include <CLHEP/Random/RandGaussQ.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

ElectronEnergyCalibratorRun2::ElectronEnergyCalibratorRun2(EpCombinationToolSemi &combinator,
							   bool isMC,
							   bool synchronization,
							   std::string correctionFile
							   ) :
  epCombinationTool_(&combinator),
  isMC_(isMC), synchronization_(synchronization),
  rng_(0),
  correctionRetriever_(correctionFile) // here is opening the files and reading the corrections
{
  if(isMC_) {
    correctionRetriever_.doScale = false;
    correctionRetriever_.doSmearings = true;
  } else {
    correctionRetriever_.doScale = true;
    correctionRetriever_.doSmearings = false;
  }
}

ElectronEnergyCalibratorRun2::~ElectronEnergyCalibratorRun2()
{}

void ElectronEnergyCalibratorRun2::initPrivateRng(TRandom *rnd)
{
  rng_ = rnd;
}

std::vector<float> ElectronEnergyCalibratorRun2::calibrate(reco::GsfElectron &electron, unsigned int runNumber, 
							   const EcalRecHitCollection *recHits, edm::StreamID const &id, int eventIsMC) const
{
  float smear = 0.0, scale = 1.0;
  float aeta = std::abs(electron.superCluster()->eta());
  float et = electron.correctedEcalEnergy() / cosh(aeta);

  if (et < 1.0) {
    return std::vector<float>(10, electron.correctedEcalEnergy());
  }

  DetId seedDetId = electron.superCluster()->seed()->seed();
  EcalRecHitCollection::const_iterator seedRecHit = recHits->find(seedDetId);
  unsigned int gainSeedSC = 12;
  if (seedRecHit != recHits->end()) { 
    if(seedRecHit->checkFlag(EcalRecHit::kHasSwitchToGain6)) gainSeedSC = 6;
    if(seedRecHit->checkFlag(EcalRecHit::kHasSwitchToGain1)) gainSeedSC = 1;
  }
  scale = correctionRetriever_.ScaleCorrection(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC);  
  smear = correctionRetriever_.getSmearingSigma(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 0., 0.);

  // This always to be carefully thought
  float scale_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 7);
  float scale_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 7);
  float resol_up  = correctionRetriever_.getSmearingSigma(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 1., 0.);
  float resol_dn  = correctionRetriever_.getSmearingSigma(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, -1., 0.);

  float scale_stat_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 1);
  float scale_stat_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 1);
  float scale_syst_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 2);
  float scale_syst_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 2);
  float scale_gain_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 4);
  float scale_gain_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 4);
  float resol_rho_up  = correctionRetriever_.getSmearingSigma(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 1., 0.);
  float resol_rho_dn  = correctionRetriever_.getSmearingSigma(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, -1., 0.);
  float resol_phi_up  = correctionRetriever_.getSmearingSigma(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 0., 1.);
  float resol_phi_dn  = correctionRetriever_.getSmearingSigma(runNumber, electron.isEB(), electron.full5x5_r9(), aeta, et, gainSeedSC, 0., -1.);
  std::vector<float> uncertainties;

  double newEcalEnergy = electron.correctedEcalEnergy();
  double newEcalEnergyError = electron.correctedEcalEnergyError();
  std::pair<float, float> combinedMomentum;

  if ((eventIsMC < 0 && isMC_) || (eventIsMC == 1)) {
    double rndm = gauss(id);
    double corr = 1.0 + smear * rndm;
    double corr_rho_up = 1.0 + resol_rho_up * rndm;
    double corr_rho_dn = 1.0 + resol_rho_dn * rndm;
    double corr_phi_up = 1.0 + resol_phi_up * rndm;
    double corr_phi_dn = 1.0 + resol_phi_dn * rndm;

    double corr_up = 1.0 + resol_phi_up * rndm;
    double corr_dn = 1.0 + resol_phi_dn * rndm;

    electron.setCorrectedEcalEnergy(newEcalEnergy * corr);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * corr, smear * newEcalEnergy * corr));
    combinedMomentum = epCombinationTool_->combine(electron);

    uncertainties.push_back(combinedMomentum.first);
    uncertainties.push_back(combinedMomentum.first);
    uncertainties.push_back(combinedMomentum.first);
    uncertainties.push_back(combinedMomentum.first);
    uncertainties.push_back(combinedMomentum.first);
    uncertainties.push_back(combinedMomentum.first);
    
    electron.setCorrectedEcalEnergy(newEcalEnergy * corr_rho_up);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * corr_rho_up, resol_rho_up * newEcalEnergy * corr_rho_up));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);
    
    electron.setCorrectedEcalEnergy(newEcalEnergy * corr_rho_dn);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * corr_rho_dn, resol_rho_dn * newEcalEnergy * corr_rho_dn));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);
    
    electron.setCorrectedEcalEnergy(newEcalEnergy * corr_phi_up);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * corr_phi_up, resol_phi_up * newEcalEnergy * corr_phi_up));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);
    
    electron.setCorrectedEcalEnergy(newEcalEnergy * corr_phi_dn);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * corr_phi_dn, resol_phi_dn * newEcalEnergy * corr_phi_dn));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    // The total variation
    uncertainties.push_back(combinedMomentum.first);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * corr_up);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * corr_up, resol_up * newEcalEnergy * corr_up));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);
    
    electron.setCorrectedEcalEnergy(newEcalEnergy * corr_dn);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * corr_dn, resol_dn * newEcalEnergy * corr_dn));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);
    
    newEcalEnergy      = newEcalEnergy * corr;
    newEcalEnergyError = std::hypot(newEcalEnergyError * corr, smear * newEcalEnergy);
    electron.setCorrectedEcalEnergy(newEcalEnergy);
    electron.setCorrectedEcalEnergyError(newEcalEnergyError);
    combinedMomentum = epCombinationTool_->combine(electron);

  } else if ((eventIsMC < 0 && !isMC_) || (eventIsMC == 0)) {

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale_stat_up);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale_stat_up, smear * newEcalEnergy * scale_stat_up));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale_stat_dn);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale_stat_dn, smear * newEcalEnergy * scale_stat_dn));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale_syst_up);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale_syst_up, smear * newEcalEnergy * scale_syst_up));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale_syst_dn);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale_syst_dn, smear * newEcalEnergy * scale_syst_dn));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale_gain_up);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale_gain_up, smear * newEcalEnergy * scale_gain_up));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale_gain_dn);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale_gain_dn, smear * newEcalEnergy * scale_gain_dn));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale, resol_rho_up * newEcalEnergy * scale));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale, resol_rho_dn * newEcalEnergy * scale));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale, resol_phi_up * newEcalEnergy * scale));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale, resol_phi_dn * newEcalEnergy * scale));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    // The total variation
    electron.setCorrectedEcalEnergy(newEcalEnergy * scale_up);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale_up, smear * newEcalEnergy * scale_up));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale_dn);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale_dn, smear * newEcalEnergy * scale_dn));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale, resol_up * newEcalEnergy * scale));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    electron.setCorrectedEcalEnergy(newEcalEnergy * scale);
    electron.setCorrectedEcalEnergyError(std::hypot(newEcalEnergyError * scale, resol_dn * newEcalEnergy * scale));
    combinedMomentum = epCombinationTool_->combine(electron);
    uncertainties.push_back(combinedMomentum.first);

    newEcalEnergy      = newEcalEnergy * scale;
    newEcalEnergyError = std::hypot(newEcalEnergyError * scale, smear * newEcalEnergy);
    electron.setCorrectedEcalEnergy(newEcalEnergy);
    electron.setCorrectedEcalEnergyError(newEcalEnergyError);
    combinedMomentum = epCombinationTool_->combine(electron);

  }
  
  math::XYZTLorentzVector oldFourMomentum = electron.p4();
  math::XYZTLorentzVector newFourMomentum = math::XYZTLorentzVector(oldFourMomentum.x() * combinedMomentum.first / oldFourMomentum.t(),
								    oldFourMomentum.y() * combinedMomentum.first / oldFourMomentum.t(),
								    oldFourMomentum.z() * combinedMomentum.first / oldFourMomentum.t(),
								    combinedMomentum.first);
  electron.correctMomentum(newFourMomentum, electron.trackMomentumError(), combinedMomentum.second);  
  return uncertainties;
  
}

double ElectronEnergyCalibratorRun2::gauss(edm::StreamID const& id) const
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

