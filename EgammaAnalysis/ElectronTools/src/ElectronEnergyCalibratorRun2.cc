#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibratorRun2.h"
#include <CLHEP/Random/RandGaussQ.h>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "EgammaAnalysis/ElectronTools/interface/EGEnergySysIndex.h"

ElectronEnergyCalibratorRun2::ElectronEnergyCalibratorRun2(EpCombinationToolSemi &combinator,
							   bool isMC,
							   bool synchronization,
							   std::string correctionFile
							   ) :
  epCombinationTool_(&combinator),
  isMC_(isMC), synchronization_(synchronization),
  rng_(nullptr),
  minEt_(1.0),
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

std::vector<float> ElectronEnergyCalibratorRun2::calibrate(reco::GsfElectron &ele,
							   const unsigned int runNumber, 
							   const EcalRecHitCollection *recHits, 
							   edm::StreamID const & id, 
							   const int eventIsMC) const
{
  return calibrate(ele,runNumber,recHits,gauss(id),eventIsMC);
}

std::vector<float> ElectronEnergyCalibratorRun2::calibrate(reco::GsfElectron &ele, unsigned int runNumber, 
							   const EcalRecHitCollection *recHits, 
							   const float smearNrSigma, 
							   const int eventIsMC) const
{
  const float aeta = std::abs(ele.superCluster()->eta());
  const float et = ele.ecalEnergy() / cosh(aeta);

  if (et < minEt_) {
    std::vector<float> retVal(EGEnergySysIndex::kNrSysErrs,ele.energy());
    retVal[EGEnergySysIndex::kScaleValue]  = 1.0;
    retVal[EGEnergySysIndex::kSmearValue]  = 0.0;
    retVal[EGEnergySysIndex::kSmearNrSigma] = smearNrSigma;
    retVal[EGEnergySysIndex::kEcalPreCorr] = ele.ecalEnergy();
    retVal[EGEnergySysIndex::kEcalErrPreCorr] = ele.ecalEnergyError();
    retVal[EGEnergySysIndex::kEcalPostCorr] = ele.ecalEnergy();
    retVal[EGEnergySysIndex::kEcalErrPostCorr] = ele.ecalEnergyError();
    retVal[EGEnergySysIndex::kEcalTrkPreCorr] = ele.energy();
    retVal[EGEnergySysIndex::kEcalTrkErrPreCorr] = ele.corrections().combinedP4Error;
    retVal[EGEnergySysIndex::kEcalTrkPostCorr] = ele.energy();
    retVal[EGEnergySysIndex::kEcalTrkErrPostCorr] = ele.corrections().combinedP4Error;
    return retVal;
  }

  const DetId seedDetId = ele.superCluster()->seed()->seed();
  EcalRecHitCollection::const_iterator seedRecHit = recHits->find(seedDetId);
  unsigned int gainSeedSC = 12;
  if (seedRecHit != recHits->end()) { 
    if(seedRecHit->checkFlag(EcalRecHit::kHasSwitchToGain6)) gainSeedSC = 6;
    if(seedRecHit->checkFlag(EcalRecHit::kHasSwitchToGain1)) gainSeedSC = 1;
  }
  const float scale = correctionRetriever_.ScaleCorrection(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC);  
  const float smear = correctionRetriever_.getSmearingSigma(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 0., 0.);

  // This always to be carefully thought
  float scale_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 7);
  float scale_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 7);
  float resol_up  = correctionRetriever_.getSmearingSigma(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 1., 0.);
  float resol_dn  = correctionRetriever_.getSmearingSigma(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, -1., 0.);

  float scale_stat_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 1);
  float scale_stat_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 1);
  float scale_syst_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 2);
  float scale_syst_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 2);
  float scale_gain_up = scale + correctionRetriever_.ScaleCorrectionUncertainty(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 4);
  float scale_gain_dn = scale - correctionRetriever_.ScaleCorrectionUncertainty(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 4);
  float resol_rho_up  = correctionRetriever_.getSmearingSigma(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 1., 0.);
  float resol_rho_dn  = correctionRetriever_.getSmearingSigma(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, -1., 0.);
  float resol_phi_up  = correctionRetriever_.getSmearingSigma(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 0., 1.);
  float resol_phi_dn  = correctionRetriever_.getSmearingSigma(runNumber, ele.isEB(), ele.full5x5_r9(), aeta, et, gainSeedSC, 0., -1.);

  std::vector<float> uncertainties(EGEnergySysIndex::kNrSysErrs,0.);
 
  std::pair<float, float> combinedMomentum;

  
  uncertainties[EGEnergySysIndex::kScaleValue]  = scale;
  uncertainties[EGEnergySysIndex::kSmearValue]  = smear;
  uncertainties[EGEnergySysIndex::kSmearNrSigma]  = smearNrSigma;
  uncertainties[EGEnergySysIndex::kEcalTrkPreCorr] = ele.energy();
  uncertainties[EGEnergySysIndex::kEcalTrkErrPreCorr] = ele.corrections().combinedP4Error;
  uncertainties[EGEnergySysIndex::kEcalPreCorr] = ele.ecalEnergy();
  uncertainties[EGEnergySysIndex::kEcalErrPreCorr] = ele.ecalEnergyError();
  

  if ((eventIsMC < 0 && isMC_) || (eventIsMC == 1)) {
    double corr = 1.0 + smear * smearNrSigma;
    double corr_rho_up = 1.0 + resol_rho_up * smearNrSigma;
    double corr_rho_dn = 1.0 + resol_rho_dn * smearNrSigma;
    double corr_phi_up = 1.0 + resol_phi_up * smearNrSigma;
    double corr_phi_dn = 1.0 + resol_phi_dn * smearNrSigma;

    double corr_up = 1.0 + resol_phi_up * smearNrSigma;
    double corr_dn = 1.0 + resol_phi_dn * smearNrSigma;

    uncertainties[EGEnergySysIndex::kScaleStatUp]   = calCombinedMom(ele,scale_stat_up*corr,smear).first;
    uncertainties[EGEnergySysIndex::kScaleStatDown] = calCombinedMom(ele,scale_stat_dn*corr,smear).first;
    uncertainties[EGEnergySysIndex::kScaleSystUp]   = calCombinedMom(ele,scale_syst_up*corr,smear).first;
    uncertainties[EGEnergySysIndex::kScaleSystDown] = calCombinedMom(ele,scale_syst_dn*corr,smear).first;
    uncertainties[EGEnergySysIndex::kScaleGainUp]   = calCombinedMom(ele,scale_gain_up*corr,smear).first;
    uncertainties[EGEnergySysIndex::kScaleGainDown] = calCombinedMom(ele,scale_gain_up*corr,smear).first;
    
    uncertainties[EGEnergySysIndex::kSmearRhoUp]   = calCombinedMom(ele,corr_rho_up,resol_rho_up).first;
    uncertainties[EGEnergySysIndex::kSmearRhoDown] = calCombinedMom(ele,corr_rho_dn,resol_rho_dn).first;
    uncertainties[EGEnergySysIndex::kSmearPhiUp]   = calCombinedMom(ele,corr_phi_up,resol_phi_up).first;
    uncertainties[EGEnergySysIndex::kSmearPhiDown] = calCombinedMom(ele,corr_phi_dn,resol_phi_dn).first;

    uncertainties[EGEnergySysIndex::kScaleUp]   = calCombinedMom(ele,scale_up*corr,smear).first;
    uncertainties[EGEnergySysIndex::kScaleDown] = calCombinedMom(ele,scale_dn*corr,smear).first;
    uncertainties[EGEnergySysIndex::kSmearUp]   = calCombinedMom(ele,corr_up,resol_up).first;
    uncertainties[EGEnergySysIndex::kSmearDown] = calCombinedMom(ele,corr_dn,resol_dn).first;


    setEcalEnergy(ele,corr,smear);
    combinedMomentum = epCombinationTool_->combine(ele);
    
  } else if ((eventIsMC < 0 && !isMC_) || (eventIsMC == 0)) {
    
    uncertainties[EGEnergySysIndex::kScaleStatUp]   = calCombinedMom(ele,scale_stat_up,smear).first;
    uncertainties[EGEnergySysIndex::kScaleStatDown] = calCombinedMom(ele,scale_stat_dn,smear).first;
    uncertainties[EGEnergySysIndex::kScaleSystUp]   = calCombinedMom(ele,scale_syst_up,smear).first;
    uncertainties[EGEnergySysIndex::kScaleSystDown] = calCombinedMom(ele,scale_syst_dn,smear).first;
    uncertainties[EGEnergySysIndex::kScaleGainUp]   = calCombinedMom(ele,scale_gain_up,smear).first;
    uncertainties[EGEnergySysIndex::kScaleGainDown] = calCombinedMom(ele,scale_gain_dn,smear).first;

    uncertainties[EGEnergySysIndex::kSmearRhoUp]    = calCombinedMom(ele,scale,resol_rho_up).first;
    uncertainties[EGEnergySysIndex::kSmearRhoDown]  = calCombinedMom(ele,scale,resol_rho_dn).first;
    uncertainties[EGEnergySysIndex::kSmearPhiUp]    = calCombinedMom(ele,scale,resol_phi_up).first;
    uncertainties[EGEnergySysIndex::kSmearPhiDown]  = calCombinedMom(ele,scale,resol_phi_dn).first;

    uncertainties[EGEnergySysIndex::kScaleUp]       = calCombinedMom(ele,scale_up,smear).first;
    uncertainties[EGEnergySysIndex::kScaleDown]     = calCombinedMom(ele,scale_dn,smear).first;
    uncertainties[EGEnergySysIndex::kSmearUp]       = calCombinedMom(ele,scale,resol_phi_up).first;
    uncertainties[EGEnergySysIndex::kSmearDown]     = calCombinedMom(ele,scale,resol_phi_dn).first;
    
    setEcalEnergy(ele,scale,smear);
    combinedMomentum = epCombinationTool_->combine(ele);

  }
  
  math::XYZTLorentzVector oldFourMomentum = ele.p4();
  math::XYZTLorentzVector newFourMomentum = math::XYZTLorentzVector(oldFourMomentum.x() * combinedMomentum.first / oldFourMomentum.t(),
								    oldFourMomentum.y() * combinedMomentum.first / oldFourMomentum.t(),
								    oldFourMomentum.z() * combinedMomentum.first / oldFourMomentum.t(),
								    combinedMomentum.first);
  ele.correctMomentum(newFourMomentum, ele.trackMomentumError(), combinedMomentum.second); 
  uncertainties[EGEnergySysIndex::kEcalTrkPostCorr] = ele.energy();
  uncertainties[EGEnergySysIndex::kEcalTrkErrPostCorr] = ele.corrections().combinedP4Error;
 
  uncertainties[EGEnergySysIndex::kEcalPostCorr] = ele.ecalEnergy();
  uncertainties[EGEnergySysIndex::kEcalErrPostCorr] = ele.ecalEnergyError();
 
 
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

void ElectronEnergyCalibratorRun2::setEcalEnergy(reco::GsfElectron& ele,
						 const float scale,
						 const float smear)const
{
  const float oldEcalEnergy = ele.ecalEnergy();
  const float oldEcalEnergyErr = ele.ecalEnergyError();
  ele.setCorrectedEcalEnergy( oldEcalEnergy*scale );
  ele.setCorrectedEcalEnergyError(std::hypot( oldEcalEnergyErr*scale, oldEcalEnergy*smear*scale ) );
}

std::pair<float,float> ElectronEnergyCalibratorRun2::calCombinedMom(reco::GsfElectron& ele,
								    const float scale,
								    const float smear)const
{ 
  const float oldEcalEnergy = ele.ecalEnergy();
  const float oldEcalEnergyErr = ele.ecalEnergyError();
  setEcalEnergy(ele,scale,smear);
  const auto& combinedMomentum = epCombinationTool_->combine(ele);
  ele.setCorrectedEcalEnergy(oldEcalEnergy);
  ele.setCorrectedEcalEnergyError(oldEcalEnergyErr);
  return combinedMomentum;
}
						      
