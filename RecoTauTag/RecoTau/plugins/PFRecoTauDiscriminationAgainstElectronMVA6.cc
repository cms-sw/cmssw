/* class PFRecoTauDiscriminationAgainstElectronMVA6
 * created : Nov 2 2015,
 * revised : ,
 * Authorss : Fabio Colombo (KIT)
 */

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA6.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <iostream>
#include <sstream>
#include <fstream>

using namespace reco;

class PFRecoTauDiscriminationAgainstElectronMVA6 : public PFTauDiscriminationContainerProducerBase {
public:
  explicit PFRecoTauDiscriminationAgainstElectronMVA6(const edm::ParameterSet& cfg)
      : PFTauDiscriminationContainerProducerBase(cfg), mva_() {
    mva_ = std::make_unique<AntiElectronIDMVA6>(cfg);

    srcGsfElectrons_ = cfg.getParameter<edm::InputTag>("srcGsfElectrons");
    GsfElectrons_token = consumes<reco::GsfElectronCollection>(srcGsfElectrons_);
    vetoEcalCracks_ = cfg.getParameter<bool>("vetoEcalCracks");

    verbosity_ = cfg.getParameter<int>("verbosity");
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&) override;

  reco::SingleTauDiscriminatorContainer discriminate(const PFTauRef&) const override;

  ~PFRecoTauDiscriminationAgainstElectronMVA6() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool isInEcalCrack(double) const;

  std::string moduleLabel_;
  std::unique_ptr<AntiElectronIDMVA6> mva_;

  edm::InputTag srcGsfElectrons_;
  edm::EDGetTokenT<reco::GsfElectronCollection> GsfElectrons_token;
  edm::Handle<reco::GsfElectronCollection> gsfElectrons_;
  edm::Handle<TauCollection> taus_;

  bool vetoEcalCracks_;

  int verbosity_;
};

void PFRecoTauDiscriminationAgainstElectronMVA6::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
  mva_->beginEvent(evt, es);

  evt.getByToken(Tau_token, taus_);

  evt.getByToken(GsfElectrons_token, gsfElectrons_);
}

reco::SingleTauDiscriminatorContainer PFRecoTauDiscriminationAgainstElectronMVA6::discriminate(
    const PFTauRef& thePFTauRef) const {
  reco::SingleTauDiscriminatorContainer result;
  result.rawValues = {1., -1.};
  double category = -1.;
  bool isGsfElectronMatched = false;

  float deltaRDummy = 9.9;

  const float ECALBarrelEndcapEtaBorder = 1.479;
  float tauEtaAtEcalEntrance = -99.;
  float sumEtaTimesEnergy = 0.;
  float sumEnergy = 0.;
  for (const auto& pfCandidate : thePFTauRef->signalPFCands()) {
    sumEtaTimesEnergy += (pfCandidate->positionAtECALEntrance().eta() * pfCandidate->energy());
    sumEnergy += pfCandidate->energy();
  }
  if (sumEnergy > 0.) {
    tauEtaAtEcalEntrance = sumEtaTimesEnergy / sumEnergy;
  }

  float leadChargedPFCandEtaAtEcalEntrance = -99.;
  float leadChargedPFCandPt = -99.;
  for (const auto& pfCandidate : thePFTauRef->signalPFCands()) {
    const reco::Track* track = nullptr;
    if (pfCandidate->trackRef().isNonnull())
      track = pfCandidate->trackRef().get();
    else if (pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->innerTrack().isNonnull())
      track = pfCandidate->muonRef()->innerTrack().get();
    else if (pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->globalTrack().isNonnull())
      track = pfCandidate->muonRef()->globalTrack().get();
    else if (pfCandidate->muonRef().isNonnull() && pfCandidate->muonRef()->outerTrack().isNonnull())
      track = pfCandidate->muonRef()->outerTrack().get();
    else if (pfCandidate->gsfTrackRef().isNonnull())
      track = pfCandidate->gsfTrackRef().get();
    if (track) {
      if (track->pt() > leadChargedPFCandPt) {
        leadChargedPFCandEtaAtEcalEntrance = pfCandidate->positionAtECALEntrance().eta();
        leadChargedPFCandPt = track->pt();
      }
    }
  }

  if ((*thePFTauRef).leadChargedHadrCand().isNonnull()) {
    int numSignalGammaCandsInSigCone = 0;
    const std::vector<reco::CandidatePtr>& signalGammaCands = thePFTauRef->signalGammaCands();

    for (const auto& pfGamma : signalGammaCands) {
      double dR = deltaR(pfGamma->p4(), thePFTauRef->leadChargedHadrCand()->p4());
      double signalrad = std::max(0.05, std::min(0.10, 3.0 / std::max(1.0, thePFTauRef->pt())));

      // pfGammas inside the tau signal cone
      if (dR < signalrad) {
        numSignalGammaCandsInSigCone += 1;
      }
    }

    // loop over the electrons
    for (const auto& theGsfElectron : *gsfElectrons_) {
      if (theGsfElectron.pt() > 10.) {  // CV: only take electrons above some minimal energy/Pt into account...
        double deltaREleTau = deltaR(theGsfElectron.p4(), thePFTauRef->p4());
        deltaRDummy = deltaREleTau;
        if (deltaREleTau < 0.3) {
          double mva_match = mva_->MVAValue(*thePFTauRef, theGsfElectron);
          const reco::PFCandidatePtr& lpfch = thePFTauRef->leadPFChargedHadrCand();
          bool hasGsfTrack = false;
          if (lpfch.isNonnull()) {
            hasGsfTrack = lpfch->gsfTrackRef().isNonnull();
          }
          if (!hasGsfTrack)
            hasGsfTrack = theGsfElectron.gsfTrack().isNonnull();

          //// Veto taus that go to Ecal crack
          if (vetoEcalCracks_ &&
              (isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance))) {
            // return MVA output value
            result.rawValues.at(0) = -99.;
            return result;
          }
          //// Veto taus that go to Ecal crack

          if (std::abs(tauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder) {  // Barrel
            if (numSignalGammaCandsInSigCone == 0 && hasGsfTrack) {
              category = 5.;
            } else if (numSignalGammaCandsInSigCone >= 1 && hasGsfTrack) {
              category = 7.;
            }
          } else {  // Endcap
            if (numSignalGammaCandsInSigCone == 0 && hasGsfTrack) {
              category = 13.;
            } else if (numSignalGammaCandsInSigCone >= 1 && hasGsfTrack) {
              category = 15.;
            }
          }

          result.rawValues.at(0) = std::min(result.rawValues.at(0), float(mva_match));
          isGsfElectronMatched = true;
        }  // deltaR < 0.3
      }    // electron pt > 10
    }      // end of loop over electrons

    if (!isGsfElectronMatched) {
      result.rawValues.at(0) = mva_->MVAValue(*thePFTauRef);
      const reco::PFCandidatePtr& lpfch = thePFTauRef->leadPFChargedHadrCand();
      bool hasGsfTrack = false;
      if (lpfch.isNonnull()) {
        hasGsfTrack = lpfch->gsfTrackRef().isNonnull();
      }

      //// Veto taus that go to Ecal crack
      if (vetoEcalCracks_ &&
          (isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance))) {
        // add category index
        result.rawValues.at(1) = category;
        // return MVA output value
        result.rawValues.at(0) = -99.;
        return result;
      }
      //// Veto taus that go to Ecal crack

      if (std::abs(tauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder) {  // Barrel
        if (numSignalGammaCandsInSigCone == 0 && !hasGsfTrack) {
          category = 0.;
        } else if (numSignalGammaCandsInSigCone >= 1 && !hasGsfTrack) {
          category = 2.;
        }
      } else {  // Endcap
        if (numSignalGammaCandsInSigCone == 0 && !hasGsfTrack) {
          category = 8.;
        } else if (numSignalGammaCandsInSigCone >= 1 && !hasGsfTrack) {
          category = 10.;
        }
      }
    }
  }

  if (verbosity_) {
    edm::LogPrint("PFTauAgainstEleMVA6") << "<PFRecoTauDiscriminationAgainstElectronMVA6::discriminate>:";
    edm::LogPrint("PFTauAgainstEleMVA6") << " tau: Pt = " << thePFTauRef->pt() << ", eta = " << thePFTauRef->eta()
                                         << ", phi = " << thePFTauRef->phi();
    edm::LogPrint("PFTauAgainstEleMVA6") << " deltaREleTau = " << deltaRDummy
                                         << ", isGsfElectronMatched = " << isGsfElectronMatched;
    edm::LogPrint("PFTauAgainstEleMVA6") << " #Prongs = " << thePFTauRef->signalChargedHadrCands().size();
    edm::LogPrint("PFTauAgainstEleMVA6") << " MVA = " << result.rawValues.at(0) << ", category = " << category;
  }

  // add category index
  result.rawValues.at(1) = category;
  // return MVA output value
  return result;
}

bool PFRecoTauDiscriminationAgainstElectronMVA6::isInEcalCrack(double eta) const {
  double absEta = fabs(eta);
  return (absEta > 1.460 && absEta < 1.558);
}

void PFRecoTauDiscriminationAgainstElectronMVA6::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfRecoTauDiscriminationAgainstElectronMVA6
  edm::ParameterSetDescription desc;
  desc.add<double>("minMVANoEleMatchWOgWOgsfBL", 0.0);
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfTauProducer"));
  desc.add<double>("minMVANoEleMatchWgWOgsfBL", 0.0);
  desc.add<std::string>("mvaName_wGwGSF_EC", "gbr_wGwGSF_EC");
  desc.add<double>("minMVAWgWgsfBL", 0.0);
  desc.add<std::string>("mvaName_woGwGSF_EC", "gbr_woGwGSF_EC");
  desc.add<double>("minMVAWOgWgsfEC", 0.0);
  desc.add<std::string>("mvaName_wGwGSF_BL", "gbr_wGwGSF_BL");
  desc.add<std::string>("mvaName_woGwGSF_BL", "gbr_woGwGSF_BL");
  desc.add<bool>("returnMVA", true);
  desc.add<bool>("loadMVAfromDB", true);
  {
    edm::ParameterSetDescription pset_Prediscriminants;
    pset_Prediscriminants.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut");
      psd1.add<edm::InputTag>("Producer");
      pset_Prediscriminants.addOptional<edm::ParameterSetDescription>("leadTrack", psd1);
    }
    {
      // encountered this at
      // RecoTauTag/Configuration/python/HPSPFTaus_cff.py
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut");
      psd1.add<edm::InputTag>("Producer");
      pset_Prediscriminants.addOptional<edm::ParameterSetDescription>("decayMode", psd1);
    }
    desc.add<edm::ParameterSetDescription>("Prediscriminants", pset_Prediscriminants);
  }
  desc.add<std::string>("mvaName_NoEleMatch_woGwoGSF_BL", "gbr_NoEleMatch_woGwoGSF_BL");
  desc.add<bool>("vetoEcalCracks", true);
  desc.add<bool>("usePhiAtEcalEntranceExtrapolation", false);
  desc.add<std::string>("mvaName_NoEleMatch_wGwoGSF_BL", "gbr_NoEleMatch_wGwoGSF_BL");
  desc.add<double>("minMVANoEleMatchWOgWOgsfEC", 0.0);
  desc.add<double>("minMVAWOgWgsfBL", 0.0);
  desc.add<double>("minMVAWgWgsfEC", 0.0);
  desc.add<int>("verbosity", 0);
  desc.add<std::string>("mvaName_NoEleMatch_wGwoGSF_EC", "gbr_NoEleMatch_wGwoGSF_EC");
  desc.add<std::string>("method", "BDTG");
  desc.add<edm::InputTag>("srcGsfElectrons", edm::InputTag("gedGsfElectrons"));
  desc.add<std::string>("mvaName_NoEleMatch_woGwoGSF_EC", "gbr_NoEleMatch_woGwoGSF_EC");
  desc.add<double>("minMVANoEleMatchWgWOgsfEC", 0.0);
  descriptions.add("pfRecoTauDiscriminationAgainstElectronMVA6", desc);
}

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA6);
