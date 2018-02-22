#include "EgammaAnalysis/ElectronTools/interface/PhotonEnergyCalibratorRun2.h"
#include <CLHEP/Random/RandGaussQ.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "EgammaAnalysis/ElectronTools/interface/EGEnergySysIndex.h"

PhotonEnergyCalibratorRun2::PhotonEnergyCalibratorRun2(bool isMC, bool synchronization,
						       std::string correctionFile
						       ) :
  isMC_(isMC), synchronization_(synchronization),
  rng_(nullptr),
  minEt_(1.0),
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

std::vector<float> PhotonEnergyCalibratorRun2::calibrate(reco::Photon &photon,const unsigned int runNumber, 
							 const EcalRecHitCollection *recHits, 
							 edm::StreamID const & id, 
							 const int eventIsMC) const
{
  return calibrate(photon,runNumber,recHits,gauss(id),eventIsMC);
}

std::vector<float> PhotonEnergyCalibratorRun2::calibrate(reco::Photon &photon,const unsigned int runNumber, 
							 const EcalRecHitCollection *recHits, 
							 const float smearNrSigma, 
							 const int eventIsMC) const
{
  const float aeta = std::abs(photon.superCluster()->eta());
  const float et = photon.getCorrectedEnergy(reco::Photon::P4type::regression2) / cosh(aeta);

  if (et < minEt_) {
    std::vector<float> retVal(EGEnergySysIndex::kNrSysErrs,
			      photon.getCorrectedEnergy(reco::Photon::P4type::regression2));
    retVal[EGEnergySysIndex::kScaleValue]  = 1.0;
    retVal[EGEnergySysIndex::kSmearValue]  = 0.0;
    retVal[EGEnergySysIndex::kSmearNrSigma]  = smearNrSigma;
    retVal[EGEnergySysIndex::kEcalErrPreCorr] = photon.getCorrectedEnergyError(reco::Photon::P4type::regression2); 
    retVal[EGEnergySysIndex::kEcalErrPostCorr] = photon.getCorrectedEnergyError(reco::Photon::P4type::regression2);
    retVal[EGEnergySysIndex::kEcalTrkPreCorr] = 0.;
    retVal[EGEnergySysIndex::kEcalTrkErrPreCorr] = 0.;
    retVal[EGEnergySysIndex::kEcalTrkPostCorr] = 0.;
    retVal[EGEnergySysIndex::kEcalTrkErrPostCorr] = 0.;
    
    return retVal;
  }

  const DetId seedDetId = photon.superCluster()->seed()->seed();
  EcalRecHitCollection::const_iterator seedRecHit = recHits->find(seedDetId);
  unsigned int gainSeedSC = 12;
  if (seedRecHit != recHits->end()) {
    if(seedRecHit->checkFlag(EcalRecHit::kHasSwitchToGain6)) gainSeedSC = 6;
    if(seedRecHit->checkFlag(EcalRecHit::kHasSwitchToGain1)) gainSeedSC = 1;
  }
  const float scale = correctionRetriever_.ScaleCorrection(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC);
  const float smear = correctionRetriever_.getSmearingSigma(runNumber, photon.isEB(), photon.full5x5_r9(), aeta, et, gainSeedSC, 0., 0.);

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

  std::vector<float> uncertainties(EGEnergySysIndex::kNrSysErrs,0.);

  const double oldEcalEnergy = photon.getCorrectedEnergy(reco::Photon::P4type::regression2);
  const double oldEcalEnergyError = photon.getCorrectedEnergyError(reco::Photon::P4type::regression2);
  
  uncertainties[EGEnergySysIndex::kScaleValue]  = scale;
  uncertainties[EGEnergySysIndex::kSmearValue]  = smear;
  uncertainties[EGEnergySysIndex::kSmearNrSigma]  = smearNrSigma;
  uncertainties[EGEnergySysIndex::kEcalPreCorr] = oldEcalEnergy;
  uncertainties[EGEnergySysIndex::kEcalErrPreCorr] = oldEcalEnergyError;
 

  if ((eventIsMC < 0 && isMC_) || (eventIsMC == 1)) {
    double corr = 1.0 + smear * smearNrSigma;
    double corr_rho_up = 1.0 + resol_rho_up * smearNrSigma;
    double corr_rho_dn = 1.0 + resol_rho_dn * smearNrSigma;
    double corr_phi_up = 1.0 + resol_phi_up * smearNrSigma;
    double corr_phi_dn = 1.0 + resol_phi_dn * smearNrSigma;

    double corr_up = 1.0 + resol_up * smearNrSigma;
    double corr_dn = 1.0 + resol_dn * smearNrSigma;

    const double newEcalEnergy      = oldEcalEnergy * corr;
    const double newEcalEnergyError = std::hypot(oldEcalEnergyError * corr, smear * newEcalEnergy);
    photon.setCorrectedEnergy(reco::Photon::P4type::regression2, newEcalEnergy, newEcalEnergyError, true);
 
    uncertainties[EGEnergySysIndex::kScaleStatUp]   = newEcalEnergy * scale_stat_up/scale;
    uncertainties[EGEnergySysIndex::kScaleStatDown] = newEcalEnergy * scale_stat_dn/scale;
    uncertainties[EGEnergySysIndex::kScaleSystUp]   = newEcalEnergy * scale_syst_up/scale;
    uncertainties[EGEnergySysIndex::kScaleSystDown] = newEcalEnergy * scale_syst_dn/scale;
    uncertainties[EGEnergySysIndex::kScaleGainUp]   = newEcalEnergy * scale_gain_up/scale;
    uncertainties[EGEnergySysIndex::kScaleGainDown] = newEcalEnergy * scale_gain_dn/scale;
    uncertainties[EGEnergySysIndex::kSmearRhoUp]    = newEcalEnergy * corr_rho_up/corr;
    uncertainties[EGEnergySysIndex::kSmearRhoDown]  = newEcalEnergy * corr_rho_dn/corr;
    uncertainties[EGEnergySysIndex::kSmearPhiUp]    = newEcalEnergy * corr_phi_up/corr;
    uncertainties[EGEnergySysIndex::kSmearPhiDown]  = newEcalEnergy * corr_phi_dn/corr;
    
    // The total variation
    uncertainties[EGEnergySysIndex::kScaleUp]   = newEcalEnergy * scale_up/scale;
    uncertainties[EGEnergySysIndex::kScaleDown] = newEcalEnergy * scale_dn/scale;
    uncertainties[EGEnergySysIndex::kSmearUp]   = newEcalEnergy * corr_up/corr;
    uncertainties[EGEnergySysIndex::kSmearDown] = newEcalEnergy * corr_dn/corr;
    


  } else if ((eventIsMC < 0 && !isMC_) || (eventIsMC == 0)) {

    const double newEcalEnergy = oldEcalEnergy * scale;
    const double newEcalEnergyError = std::hypot(oldEcalEnergyError * scale, smear * newEcalEnergy);
    photon.setCorrectedEnergy(reco::Photon::P4type::regression2, newEcalEnergy, newEcalEnergyError, true);
 
    uncertainties[EGEnergySysIndex::kScaleStatUp]   = newEcalEnergy * scale_stat_up/scale;
    uncertainties[EGEnergySysIndex::kScaleStatDown] = newEcalEnergy * scale_stat_dn/scale;
    uncertainties[EGEnergySysIndex::kScaleSystUp]   = newEcalEnergy * scale_syst_up/scale;
    uncertainties[EGEnergySysIndex::kScaleSystDown] = newEcalEnergy * scale_syst_dn/scale;
    uncertainties[EGEnergySysIndex::kScaleGainUp]   = newEcalEnergy * scale_gain_up/scale;
    uncertainties[EGEnergySysIndex::kScaleGainDown] = newEcalEnergy * scale_gain_dn/scale;
    uncertainties[EGEnergySysIndex::kSmearRhoUp]    = newEcalEnergy;
    uncertainties[EGEnergySysIndex::kSmearRhoDown]  = newEcalEnergy;
    uncertainties[EGEnergySysIndex::kSmearPhiUp]    = newEcalEnergy;
    uncertainties[EGEnergySysIndex::kSmearPhiDown]  = newEcalEnergy;

    // The total variation
    uncertainties[EGEnergySysIndex::kScaleUp]   = newEcalEnergy * scale_up/scale;
    uncertainties[EGEnergySysIndex::kScaleDown] = newEcalEnergy * scale_dn/scale;
    uncertainties[EGEnergySysIndex::kSmearUp]   = newEcalEnergy;
    uncertainties[EGEnergySysIndex::kSmearDown] = newEcalEnergy;
    
  }
 
  uncertainties[EGEnergySysIndex::kEcalPostCorr] = photon.getCorrectedEnergy(reco::Photon::P4type::regression2);
  uncertainties[EGEnergySysIndex::kEcalErrPostCorr] = photon.getCorrectedEnergyError(reco::Photon::P4type::regression2);

  
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

