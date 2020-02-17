/* class PATTauDiscriminationAgainstElectronMVA6
 * created : Apr 14 2016,
 * revised : ,
 * Authorss :  Anne-Catherine Le Bihan (IPHC)
 */

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA6.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <iostream>
#include <sstream>
#include <fstream>

using namespace pat;

class PATTauDiscriminationAgainstElectronMVA6 : public PATTauDiscriminationContainerProducerBase {
public:
  explicit PATTauDiscriminationAgainstElectronMVA6(const edm::ParameterSet& cfg)
      : PATTauDiscriminationContainerProducerBase(cfg), mva_() {
    mva_ = std::make_unique<AntiElectronIDMVA6>(cfg);

    srcElectrons = cfg.getParameter<edm::InputTag>("srcElectrons");
    electronToken = consumes<pat::ElectronCollection>(srcElectrons);
    vetoEcalCracks_ = cfg.getParameter<bool>("vetoEcalCracks");
    verbosity_ = cfg.getParameter<int>("verbosity");
  }

  void beginEvent(const edm::Event&, const edm::EventSetup&) override;

  reco::SingleTauDiscriminatorContainer discriminate(const TauRef&) const override;

  ~PATTauDiscriminationAgainstElectronMVA6() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool isInEcalCrack(double) const;

  std::string moduleLabel_;
  std::unique_ptr<AntiElectronIDMVA6> mva_;

  edm::InputTag srcElectrons;
  edm::EDGetTokenT<pat::ElectronCollection> electronToken;
  edm::Handle<pat::ElectronCollection> Electrons;
  edm::Handle<TauCollection> taus_;

  bool vetoEcalCracks_;

  int verbosity_;
};

void PATTauDiscriminationAgainstElectronMVA6::beginEvent(const edm::Event& evt, const edm::EventSetup& es) {
  mva_->beginEvent(evt, es);

  evt.getByToken(Tau_token, taus_);

  evt.getByToken(electronToken, Electrons);
}

reco::SingleTauDiscriminatorContainer PATTauDiscriminationAgainstElectronMVA6::discriminate(
    const TauRef& theTauRef) const {
  reco::SingleTauDiscriminatorContainer result;
  result.rawValues = {1., -1.};
  double category = -1.;
  bool isGsfElectronMatched = false;
  float deltaRDummy = 9.9;
  const float ECALBarrelEndcapEtaBorder = 1.479;
  float tauEtaAtEcalEntrance = theTauRef->etaAtEcalEntrance();
  float leadChargedPFCandEtaAtEcalEntrance = theTauRef->etaAtEcalEntranceLeadChargedCand();

  if ((*theTauRef).leadChargedHadrCand().isNonnull()) {
    int numSignalPFGammaCandsInSigCone = 0;
    const reco::CandidatePtrVector signalGammaCands = theTauRef->signalGammaCands();
    for (const auto& gamma : signalGammaCands) {
      double dR = deltaR(gamma->p4(), theTauRef->leadChargedHadrCand()->p4());
      double signalrad = std::max(0.05, std::min(0.10, 3.0 / std::max(1.0, theTauRef->pt())));
      // gammas inside the tau signal cone
      if (dR < signalrad) {
        numSignalPFGammaCandsInSigCone += 1;
      }
    }
    // loop over the electrons
    for (const auto& theElectron : *Electrons) {
      if (theElectron.pt() > 10.) {  // CV: only take electrons above some minimal energy/Pt into account...
        double deltaREleTau = deltaR(theElectron.p4(), theTauRef->p4());
        deltaRDummy = deltaREleTau;
        if (deltaREleTau < 0.3) {
          double mva_match = mva_->MVAValue(*theTauRef, theElectron);
          bool hasGsfTrack = false;
          pat::PackedCandidate const* packedLeadTauCand =
              dynamic_cast<pat::PackedCandidate const*>(theTauRef->leadChargedHadrCand().get());
          if (abs(packedLeadTauCand->pdgId()) == 11)
            hasGsfTrack = true;
          if (!hasGsfTrack)
            hasGsfTrack = theElectron.gsfTrack().isNonnull();

          // veto taus that go to Ecal crack
          if (vetoEcalCracks_ &&
              (isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance))) {
            // return MVA output value
            result.rawValues.at(0) = -99;
            return result;
          }
          // Veto taus that go to Ecal crack
          if (std::abs(tauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder) {  // Barrel
            if (numSignalPFGammaCandsInSigCone == 0 && hasGsfTrack) {
              category = 5.;
            } else if (numSignalPFGammaCandsInSigCone >= 1 && hasGsfTrack) {
              category = 7.;
            }
          } else {  // Endcap
            if (numSignalPFGammaCandsInSigCone == 0 && hasGsfTrack) {
              category = 13.;
            } else if (numSignalPFGammaCandsInSigCone >= 1 && hasGsfTrack) {
              category = 15.;
            }
          }
          result.rawValues.at(0) = std::min(result.rawValues.at(0), float(mva_match));
          isGsfElectronMatched = true;
        }  // deltaR < 0.3
      }    // electron pt > 10
    }      // end of loop over electrons

    if (!isGsfElectronMatched) {
      result.rawValues.at(0) = mva_->MVAValue(*theTauRef);
      bool hasGsfTrack = false;
      pat::PackedCandidate const* packedLeadTauCand =
          dynamic_cast<pat::PackedCandidate const*>(theTauRef->leadChargedHadrCand().get());
      if (abs(packedLeadTauCand->pdgId()) == 11)
        hasGsfTrack = true;

      // veto taus that go to Ecal crack
      if (vetoEcalCracks_ &&
          (isInEcalCrack(tauEtaAtEcalEntrance) || isInEcalCrack(leadChargedPFCandEtaAtEcalEntrance))) {
        // add category index
        result.rawValues.at(1) = category;
        // return MVA output value
        result.rawValues.at(0) = -99;
        return result;
      }
      // veto taus that go to Ecal crack
      if (std::abs(tauEtaAtEcalEntrance) < ECALBarrelEndcapEtaBorder) {  // Barrel
        if (numSignalPFGammaCandsInSigCone == 0 && !hasGsfTrack) {
          category = 0.;
        } else if (numSignalPFGammaCandsInSigCone >= 1 && !hasGsfTrack) {
          category = 2.;
        }
      } else {  // Endcap
        if (numSignalPFGammaCandsInSigCone == 0 && !hasGsfTrack) {
          category = 8.;
        } else if (numSignalPFGammaCandsInSigCone >= 1 && !hasGsfTrack) {
          category = 10.;
        }
      }
    }
  }
  if (verbosity_) {
    edm::LogPrint("PATTauAgainstEleMVA6") << "<PATTauDiscriminationAgainstElectronMVA6::discriminate>:";
    edm::LogPrint("PATTauAgainstEleMVA6")
        << " tau: Pt = " << theTauRef->pt() << ", eta = " << theTauRef->eta() << ", phi = " << theTauRef->phi();
    edm::LogPrint("PATTauAgainstEleMVA6")
        << " deltaREleTau = " << deltaRDummy << ", isGsfElectronMatched = " << isGsfElectronMatched;
    edm::LogPrint("PATTauAgainstEleMVA6") << " #Prongs = " << theTauRef->signalChargedHadrCands().size();
    edm::LogPrint("PATTauAgainstEleMVA6") << " MVA = " << result.rawValues.at(0) << ", category = " << category;
  }
  // add category index
  result.rawValues.at(1) = category;
  // return MVA output value
  return result;
}

bool PATTauDiscriminationAgainstElectronMVA6::isInEcalCrack(double eta) const {
  double absEta = fabs(eta);
  return (absEta > 1.460 && absEta < 1.558);
}

void PATTauDiscriminationAgainstElectronMVA6::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // patTauDiscriminationAgainstElectronMVA6
  edm::ParameterSetDescription desc;
  desc.add<double>("minMVANoEleMatchWOgWOgsfBL", 0.0);
  desc.add<double>("minMVANoEleMatchWgWOgsfBL", 0.0);
  desc.add<bool>("vetoEcalCracks", true);
  desc.add<bool>("usePhiAtEcalEntranceExtrapolation", false);
  desc.add<std::string>("mvaName_wGwGSF_EC", "gbr_wGwGSF_EC");
  desc.add<double>("minMVAWgWgsfBL", 0.0);
  desc.add<std::string>("mvaName_woGwGSF_EC", "gbr_woGwGSF_EC");
  desc.add<double>("minMVAWOgWgsfEC", 0.0);
  desc.add<std::string>("mvaName_wGwGSF_BL", "gbr_wGwGSF_BL");
  desc.add<std::string>("mvaName_woGwGSF_BL", "gbr_woGwGSF_BL");
  desc.add<bool>("returnMVA", true);
  desc.add<bool>("loadMVAfromDB", true);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut");
      psd1.add<edm::InputTag>("Producer");
      psd0.addOptional<edm::ParameterSetDescription>("leadTrack", psd1);
    }
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }
  desc.add<std::string>("mvaName_NoEleMatch_woGwoGSF_BL", "gbr_NoEleMatch_woGwoGSF_BL");
  desc.add<edm::InputTag>("srcElectrons", edm::InputTag("slimmedElectrons"));
  desc.add<double>("minMVANoEleMatchWOgWOgsfEC", 0.0);
  desc.add<std::string>("mvaName_NoEleMatch_wGwoGSF_BL", "gbr_NoEleMatch_wGwoGSF_BL");
  desc.add<edm::InputTag>("PATTauProducer", edm::InputTag("slimmedTaus"));
  desc.add<double>("minMVAWOgWgsfBL", 0.0);
  desc.add<double>("minMVAWgWgsfEC", 0.0);
  desc.add<int>("verbosity", 0);
  desc.add<std::string>("mvaName_NoEleMatch_wGwoGSF_EC", "gbr_NoEleMatch_wGwoGSF_EC");
  desc.add<std::string>("method", "BDTG");
  desc.add<std::string>("mvaName_NoEleMatch_woGwoGSF_EC", "gbr_NoEleMatch_woGwoGSF_EC");
  desc.add<double>("minMVANoEleMatchWgWOgsfEC", 0.0);
  descriptions.add("patTauDiscriminationAgainstElectronMVA6", desc);
}

DEFINE_FWK_MODULE(PATTauDiscriminationAgainstElectronMVA6);
