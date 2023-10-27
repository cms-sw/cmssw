/** \class PhotonMVABasedHaloTagger
 *   \author Shilpi Jain (University of Minnesota)
 *  * Links to the presentation: 
 1. ECAL DPG: https://indico.cern.ch/event/991261/contributions/4283096/attachments/2219229/3757719/beamHalo_31march_v1.pdf
 2. JetMET POG: https://indico.cern.ch/event/1027614/contributions/4314949/attachments/2224472/3767396/beamHalo_12April.pdf
 */

#include "RecoEgamma/PhotonIdentification/interface/PhotonMVABasedHaloTagger.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include <DataFormats/Math/interface/deltaPhi.h>
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

PhotonMVABasedHaloTagger::PhotonMVABasedHaloTagger(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC)
    : geometryToken_(iC.esConsumes()),
      ecalPFRechitThresholdsToken_(iC.esConsumes()),
      ecalClusterToolsESGetTokens_(std::move(iC)) {
  rhoLabel_ = iC.consumes<double>(conf.getParameter<edm::InputTag>("rhoLabel"));

  EBecalCollection_ = iC.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("barrelEcalRecHitCollection"));
  EEecalCollection_ = iC.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("endcapEcalRecHitCollection"));
  ESCollection_ = iC.consumes<EcalRecHitCollection>(conf.getParameter<edm::InputTag>("esRecHitCollection"));
  HBHERecHitsCollection_ = iC.consumes<HBHERecHitCollection>(conf.getParameter<edm::InputTag>("HBHERecHitsCollection"));
  recHitEThresholdHB_ = conf.getParameter<EgammaHcalIsolation::arrayHB>("recHitEThresholdHB");
  recHitEThresholdHE_ = conf.getParameter<EgammaHcalIsolation::arrayHE>("recHitEThresholdHE");

  noiseThrES_ = conf.getParameter<double>("noiseThrES");
}

double PhotonMVABasedHaloTagger::calculateMVA(const reco::Photon* pho,
                                              const GBRForest* gbr_,
                                              const edm::Event& iEvent,
                                              const edm::EventSetup& es) {
  bool isEB = pho->isEB();

  if (isEB)
    return 1.0;  /// this MVA is useful and trained only for the EE photons. For EB, there are a lot of other useful handles which can reject beam halo efficiently

  //rho handle
  double rho_ = iEvent.get(rhoLabel_);

  // Get all the RecHits
  const auto& ecalRecHitsBarrel = iEvent.get(EBecalCollection_);
  const auto& ecalRecHitsEndcap = iEvent.get(EEecalCollection_);
  const auto& esRecHits = iEvent.get(ESCollection_);
  const auto& hbheRecHits = iEvent.get(HBHERecHitsCollection_);

  //gets geometry
  pG_ = es.getHandle(geometryToken_);
  const CaloGeometry* geo = pG_.product();

  ///ECAL PF rechit thresholds
  auto const& thresholds = es.getData(ecalPFRechitThresholdsToken_);

  noZS::EcalClusterLazyTools lazyToolnoZS(
      iEvent, ecalClusterToolsESGetTokens_.get(es), EBecalCollection_, EEecalCollection_);

  ///calculate the energy weighted X, Y and Z position of the photon cluster
  if (isEB)
    calphoClusCoordinECAL(geo, pho, &thresholds, ecalRecHitsBarrel);
  else
    calphoClusCoordinECAL(geo, pho, &thresholds, ecalRecHitsEndcap);

  ///calculate the HBHE cluster position hypothesis
  calmatchedHBHECoordForBothHypothesis(geo, pho, hbheRecHits);
  calmatchedESCoordForBothHypothesis(geo, pho, esRecHits);

  ///this function works for EE only. Above ones work for EB as well in case later one wants to put a similar function for EB without returning 1

  double angle_EE_HE_samedPhi = calAngleBetweenEEAndSubDet(
      hcalClusNhits_samedPhi_,
      hcalClusX_samedPhi_,
      hcalClusY_samedPhi_,
      hcalClusZ_samedPhi_);  //essentially caculates the angle and energy variables in the two hypothesis between EE and HE

  double angle_EE_HE_samedR =
      calAngleBetweenEEAndSubDet(hcalClusNhits_samedR_, hcalClusX_samedR_, hcalClusY_samedR_, hcalClusZ_samedR_);

  double angle_EE_ES_samedPhi = calAngleBetweenEEAndSubDet(
      preshowerNhits_samedPhi_, preshowerX_samedPhi_, preshowerY_samedPhi_, preshowerZ_samedPhi_);

  double angle_EE_ES_samedR =
      calAngleBetweenEEAndSubDet(preshowerNhits_samedR_, preshowerX_samedR_, preshowerY_samedR_, preshowerZ_samedR_);

  ////set all the above calculated variables as input to the MVA

  const auto& vCov = lazyToolnoZS.localCovariances(*(pho->superCluster()->seed()));
  double spp = (std::isnan(vCov[2]) ? 0. : sqrt(vCov[2]));

  ///https://cmssdt.cern.ch/lxr/source/RecoEgamma/ElectronIdentification/src/ElectronMVAEstimator.cc

  float vars[15];

  vars[0] = preshowerE_samedPhi_;
  vars[1] = hcalClusE_samedPhi_;
  vars[2] = preshowerE_samedR_;
  vars[3] = hcalClusE_samedR_;
  vars[4] = pho->full5x5_r9();
  vars[5] = pho->superCluster()->etaWidth();
  vars[6] = pho->superCluster()->phiWidth();
  vars[7] = pho->full5x5_sigmaIetaIeta();
  vars[8] = spp;
  vars[9] = angle_EE_ES_samedR;
  vars[10] = angle_EE_HE_samedR;
  vars[11] = angle_EE_ES_samedPhi;
  vars[12] = angle_EE_HE_samedPhi;
  vars[13] = (pho->superCluster()->preshowerEnergyPlane1() + pho->superCluster()->preshowerEnergyPlane2()) /
             pho->superCluster()->rawEnergy();
  vars[14] = rho_;

  double BHmva = gbr_->GetGradBoostClassifier(vars);
  return BHmva;
}

void PhotonMVABasedHaloTagger::calphoClusCoordinECAL(const CaloGeometry* geo,
                                                     const reco::Photon* pho,
                                                     const EcalPFRecHitThresholds* thresholds,
                                                     const EcalRecHitCollection& ecalRecHits) {
  ecalClusX_ = 0;
  ecalClusY_ = 0;
  ecalClusZ_ = 0;
  ecalClusNhits_ = 0;
  ecalClusE_ = 0;

  double phoSCEta = pho->superCluster()->eta();
  double phoSCPhi = pho->superCluster()->phi();

  for (const auto& ecalrechit : ecalRecHits) {
    auto const det = ecalrechit.id();
    double rhE = ecalrechit.energy();
    const GlobalPoint& rechitPoint = geo->getPosition(det);

    double rhEta = rechitPoint.eta();
    double rhPhi = rechitPoint.phi();
    double rhX = rechitPoint.x();
    double rhY = rechitPoint.y();
    double rhZ = rechitPoint.z();

    if (thresholds == nullptr) {
      throw cms::Exception("EmptyPFRechHitThresCollection")
          << "In PhotonMVABasedHaloTagger::calphoClusCoordinECAL, EcalPFRecHitThresholds cannot be = nulptr";
    }

    float rhThres = (*thresholds)[det];

    if (rhE <= rhThres)
      continue;

    if (phoSCEta * rhEta < 0)
      continue;

    double dR2 = reco::deltaR2(rhEta, rhPhi, phoSCEta, phoSCPhi);

    if (dR2 < dr2Max_ECALClus_) {
      ecalClusX_ += rhX * rhE;
      ecalClusY_ += rhY * rhE;
      ecalClusZ_ += rhZ * rhE;
      ecalClusE_ += rhE;
      ecalClusNhits_++;
    }
  }  //for(int ih=0; ih<nMatchedEErh[ipho]; ih++)

  if (ecalClusNhits_ > 0) {  //should always be > 0 for an EM cluster
    ecalClusX_ = ecalClusX_ / ecalClusE_;
    ecalClusY_ = ecalClusY_ / ecalClusE_;
    ecalClusZ_ = ecalClusZ_ / ecalClusE_;
  }  //if(ecalClusNhits_>0)
}

void PhotonMVABasedHaloTagger::calmatchedHBHECoordForBothHypothesis(const CaloGeometry* geo,
                                                                    const reco::Photon* pho,
                                                                    const HBHERecHitCollection& HBHERecHits) {
  hcalClusX_samedPhi_ = 0;
  hcalClusY_samedPhi_ = 0;
  hcalClusZ_samedPhi_ = 0;

  hcalClusNhits_samedPhi_ = 0;
  hcalClusE_samedPhi_ = 0;

  hcalClusX_samedR_ = 0;
  hcalClusY_samedR_ = 0;
  hcalClusZ_samedR_ = 0;
  hcalClusNhits_samedR_ = 0;
  hcalClusE_samedR_ = 0;

  double phoSCEta = pho->superCluster()->eta();
  double phoSCPhi = pho->superCluster()->phi();

  // Loop over HBHERecHit's
  for (const auto& hbherechit : HBHERecHits) {
    HcalDetId det = hbherechit.id();
    const GlobalPoint& rechitPoint = geo->getPosition(det);

    double rhEta = rechitPoint.eta();
    double rhPhi = rechitPoint.phi();
    double rhX = rechitPoint.x();
    double rhY = rechitPoint.y();
    double rhZ = rechitPoint.z();
    double rhE = hbherechit.energy();

    int depth = det.depth();

    if ((det.subdet() == HcalBarrel and (depth < 1 or depth > int(recHitEThresholdHB_.size()))) or
        (det.subdet() == HcalEndcap and (depth < 1 or depth > int(recHitEThresholdHE_.size())))) {
      edm::LogWarning("PhotonMVABasedHaloTagger")
          << " hit in subdet " << det.subdet() << " has an unaccounted for depth of " << depth
          << "!! Leaving this hit!!";
      continue;
    }

    const bool goodHBe = det.subdet() == HcalBarrel and rhE > recHitEThresholdHB_[depth - 1];
    const bool goodHEe = det.subdet() == HcalEndcap and rhE > recHitEThresholdHE_[depth - 1];
    if (!(goodHBe or goodHEe))
      continue;

    if (phoSCEta * rhEta < 0)
      continue;  ///Should be on the same side of Z

    double dPhi = deltaPhi(phoSCPhi, rhPhi);

    ///only valid for the EE; this is 26 cm; hit within 3x3 of HCAL centered at the EECAL xtal
    bool isRHBehindECAL = std::abs(dPhi) < dPhiMax_HCALClus_SamePhi_;
    if (isRHBehindECAL) {
      double rho2 = pow(rhX, 2) + pow(rhY, 2);
      isRHBehindECAL &= (rho2 >= rho2Min_ECALpos_ && rho2 <= rho2Max_ECALpos_);
      if (isRHBehindECAL) {
        double dRho2 = pow(rhX - ecalClusX_, 2) + pow(rhY - ecalClusY_, 2);
        isRHBehindECAL &= dRho2 <= dRho2Max_HCALClus_SamePhi_;
        if (isRHBehindECAL) {
          hcalClusX_samedPhi_ += rhX * rhE;
          hcalClusY_samedPhi_ += rhY * rhE;
          hcalClusZ_samedPhi_ += rhZ * rhE;
          hcalClusE_samedPhi_ += rhE;
          hcalClusNhits_samedPhi_++;
        }
      }
    }  //if(rho>=31 && rho<=172)

    ///dont use hits which are just behind the ECAL in the same phi region
    if (!isRHBehindECAL) {
      double dR2 = reco::deltaR2(phoSCEta, phoSCPhi, rhEta, rhPhi);
      if (dR2 < dR2Max_HCALClus_SamePhi_) {
        hcalClusX_samedR_ += rhX * rhE;
        hcalClusY_samedR_ += rhY * rhE;
        hcalClusZ_samedR_ += rhZ * rhE;
        hcalClusE_samedR_ += rhE;
        hcalClusNhits_samedR_++;
      }
    }
  }  //for(int ih=0; ih<nMatchedHBHErh[ipho]; ih++)

  if (hcalClusNhits_samedPhi_ > 0) {
    hcalClusX_samedPhi_ = hcalClusX_samedPhi_ / hcalClusE_samedPhi_;
    hcalClusY_samedPhi_ = hcalClusY_samedPhi_ / hcalClusE_samedPhi_;
    hcalClusZ_samedPhi_ = hcalClusZ_samedPhi_ / hcalClusE_samedPhi_;
  }  //if(hcalClusNhits_samedPhi_>0)

  if (hcalClusNhits_samedR_ > 0) {
    hcalClusX_samedR_ = hcalClusX_samedR_ / hcalClusE_samedR_;
    hcalClusY_samedR_ = hcalClusY_samedR_ / hcalClusE_samedR_;
    hcalClusZ_samedR_ = hcalClusZ_samedR_ / hcalClusE_samedR_;
  }  //if(hcalClusNhits_samedR_>0)
}

void PhotonMVABasedHaloTagger::calmatchedESCoordForBothHypothesis(const CaloGeometry* geo,
                                                                  const reco::Photon* pho,
                                                                  const EcalRecHitCollection& ESRecHits) {
  preshowerX_samedPhi_ = 0;
  preshowerY_samedPhi_ = 0;
  preshowerZ_samedPhi_ = 0;
  preshowerNhits_samedPhi_ = 0;
  preshowerE_samedPhi_ = 0;

  preshowerX_samedR_ = 0;
  preshowerY_samedR_ = 0;
  preshowerZ_samedR_ = 0;
  preshowerNhits_samedR_ = 0;
  preshowerE_samedR_ = 0;

  double phoSCEta = pho->superCluster()->eta();
  double phoSCPhi = pho->superCluster()->phi();

  double tmpDiffdRho = 999;
  double matchX_samephi = -999;
  double matchY_samephi = -999;
  bool foundESRH_samephi = false;

  double tmpDiffdRho_samedR = 999;
  double matchX_samedR = -999;
  double matchY_samedR = -999;
  bool foundESRH_samedR = false;

  ///get theta and phi of the coordinates of photon
  double tan_theta = 1. / sinh(phoSCEta);

  double cos_phi = cos(phoSCPhi);
  double sin_phi = sin(phoSCPhi);

  for (const auto& esrechit : ESRecHits) {
    const GlobalPoint& rechitPoint = geo->getPosition(esrechit.id());

    double rhEta = rechitPoint.eta();
    double rhX = rechitPoint.x();
    double rhY = rechitPoint.y();
    double rhZ = rechitPoint.z();
    double rhE = esrechit.energy();

    if (phoSCEta * rhEta < 0)
      continue;

    if (rhE < noiseThrES_)
      continue;

    ////try to include RH according to the strips, 11 in X and 11 in Y
    /////////First calculate RH nearest in phi and eta to that of the photon SC

    //////same phi ----> the X and Y should be similar
    ////i.e. hit is required to be within the ----> seems better match with the data compared to 2.47
    double dRho2 = pow(rhX - ecalClusX_, 2) + pow(rhY - ecalClusY_, 2);

    if (dRho2 < tmpDiffdRho && dRho2 < dRho2Max_ESClus_) {
      tmpDiffdRho = dRho2;
      matchX_samephi = rhX;
      matchY_samephi = rhY;
      foundESRH_samephi = true;
    }

    ////////same eta
    ///calculate the expected x and y at the position of hte rechit
    double exp_ESRho = rhZ * tan_theta;
    double exp_ESX = cos_phi * exp_ESRho;
    double exp_ESY = sin_phi * exp_ESRho;

    double dRho_samedR2 = pow(rhX - exp_ESX, 2) + pow(rhY - exp_ESY, 2);

    if (dRho_samedR2 < tmpDiffdRho_samedR) {
      tmpDiffdRho_samedR = dRho_samedR2;
      matchX_samedR = rhX;
      matchY_samedR = rhY;
      foundESRH_samedR = true;
    }

  }  ///  for (const auto& esrechit : ESRecHits)

  ////Now calculate the sum in +/- 5 strips in X and y around the matched RH
  //+/5  strips mean = 5*~2mm = +/-10 mm = 1 cm

  for (const auto& esrechit : ESRecHits) {
    const GlobalPoint& rechitPoint = geo->getPosition(esrechit.id());

    double rhEta = rechitPoint.eta();
    double rhX = rechitPoint.x();
    double rhY = rechitPoint.y();
    double rhZ = rechitPoint.z();
    double rhE = esrechit.energy();

    if (phoSCEta * rhEta < 0)
      continue;
    if (rhE < noiseThrES_)
      continue;

    ///same phi
    bool isRHBehindECAL = foundESRH_samephi;
    if (isRHBehindECAL) {
      double dX_samephi = std::abs(matchX_samephi - rhX);
      double dY_samephi = std::abs(matchY_samephi - rhY);
      isRHBehindECAL &= (dX_samephi < dXY_ESClus_SamePhi_ && dY_samephi < dXY_ESClus_SamePhi_);
      if (isRHBehindECAL) {
        preshowerX_samedPhi_ += rhX * rhE;
        preshowerY_samedPhi_ += rhY * rhE;
        preshowerZ_samedPhi_ += rhZ * rhE;
        preshowerE_samedPhi_ += rhE;
        preshowerNhits_samedPhi_++;
      }
    }

    ///same dR
    if (!isRHBehindECAL && foundESRH_samedR) {
      double dX_samedR = std::abs(matchX_samedR - rhX);
      double dY_samedR = std::abs(matchY_samedR - rhY);

      if (dX_samedR < dXY_ESClus_SamedR_ && dY_samedR < dXY_ESClus_SamedR_) {
        preshowerX_samedR_ += rhX * rhE;
        preshowerY_samedR_ += rhY * rhE;
        preshowerZ_samedR_ += rhZ * rhE;
        preshowerE_samedR_ += rhE;
        preshowerNhits_samedR_++;
      }
    }
  }  ///for(int ih=0; ih<nMatchedESrh[ipho]; ih++)

  if (preshowerNhits_samedPhi_ > 0) {
    preshowerX_samedPhi_ = preshowerX_samedPhi_ / preshowerE_samedPhi_;
    preshowerY_samedPhi_ = preshowerY_samedPhi_ / preshowerE_samedPhi_;
    preshowerZ_samedPhi_ = preshowerZ_samedPhi_ / preshowerE_samedPhi_;
  }  //if(preshowerNhits_samedPhi_>0)

  if (preshowerNhits_samedR_ > 0) {
    preshowerX_samedR_ = preshowerX_samedR_ / preshowerE_samedR_;
    preshowerY_samedR_ = preshowerY_samedR_ / preshowerE_samedR_;
    preshowerZ_samedR_ = preshowerZ_samedR_ / preshowerE_samedR_;
  }  //if(preshowerNhits_samedR_>0)
}

double PhotonMVABasedHaloTagger::calAngleBetweenEEAndSubDet(int nhits,
                                                            double subdetClusX,
                                                            double subdetClusY,
                                                            double subdetClusZ) {
  ////get the angle of the line joining the ECAL cluster and the subdetector wrt Z axis for any hypothesis

  double angle = -999;

  if (ecalClusNhits_ > 0 && nhits > 0) {
    double dR =
        sqrt(pow(subdetClusX - ecalClusX_, 2) + pow(subdetClusY - ecalClusY_, 2) + pow(subdetClusZ - ecalClusZ_, 2));

    double cosTheta = std::abs(subdetClusZ - ecalClusZ_) / dR;

    angle = acos(cosTheta);
  }

  return angle;
}
