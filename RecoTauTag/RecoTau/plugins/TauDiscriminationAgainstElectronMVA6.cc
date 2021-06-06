/* class TauDiscriminationAgainstElectronMVA6
 * created : Nov 2 2015,
 * revised : May 29 2020,
 * Authors : Fabio Colombo (KIT)
 *           Anne-Catherine Le Bihan (IPHC),
 *           Michal Bluj (NCBJ)
 */

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include "RecoTauTag/RecoTau/interface/AntiElectronIDMVA6.h"
#include "RecoTauTag/RecoTau/interface/PositionAtECalEntranceComputer.h"
#include "DataFormats/Math/interface/deltaR.h"

template <class TauType, class TauDiscriminator, class ElectronType>
class TauDiscriminationAgainstElectronMVA6 : public TauDiscriminationProducerBase<TauType,
                                                                                  reco::TauDiscriminatorContainer,
                                                                                  reco::SingleTauDiscriminatorContainer,
                                                                                  TauDiscriminator> {
public:
  typedef std::vector<TauType> TauCollection;
  typedef edm::Ref<TauCollection> TauRef;
  typedef std::vector<ElectronType> ElectronCollection;

  explicit TauDiscriminationAgainstElectronMVA6(const edm::ParameterSet& cfg)
      : TauDiscriminationProducerBase<TauType,
                                      reco::TauDiscriminatorContainer,
                                      reco::SingleTauDiscriminatorContainer,
                                      TauDiscriminator>::TauDiscriminationProducerBase(cfg),
        moduleLabel_(cfg.getParameter<std::string>("@module_label")),
        mva_(
            std::make_unique<AntiElectronIDMVA6<TauType, ElectronType>>(cfg, edm::EDConsumerBase::consumesCollector())),
        Electron_token(edm::EDConsumerBase::consumes<ElectronCollection>(
            cfg.getParameter<edm::InputTag>("srcElectrons"))),  // MB: full specification with prefix mandatory
        positionAtECalEntrance_(PositionAtECalEntranceComputer(edm::EDConsumerBase::consumesCollector(),
                                                               cfg.getParameter<bool>("isPhase2"))),
        vetoEcalCracks_(cfg.getParameter<bool>("vetoEcalCracks")),
        isPhase2_(cfg.getParameter<bool>("isPhase2")),
        verbosity_(cfg.getParameter<int>("verbosity")) {
    deltaREleTauMax_ = (isPhase2_ ? 0.2 : 0.3);
  }

  void beginEvent(const edm::Event& evt, const edm::EventSetup& es) override {
    mva_->beginEvent(evt, es);
    positionAtECalEntrance_.beginEvent(es);
    evt.getByToken(this->Tau_token, taus_);
    evt.getByToken(Electron_token, electrons_);
  }

  reco::SingleTauDiscriminatorContainer discriminate(const TauRef&) const override;

  ~TauDiscriminationAgainstElectronMVA6() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool isInEcalCrack(double) const;

  // Overloaded method with explicit type specification to avoid partial
  //implementation of full class
  std::pair<float, float> getTauEtaAtECalEntrance(const reco::PFTauRef& theTauRef) const;
  std::pair<float, float> getTauEtaAtECalEntrance(const pat::TauRef& theTauRef) const;

  std::string moduleLabel_;
  std::unique_ptr<AntiElectronIDMVA6<TauType, ElectronType>> mva_;

  edm::EDGetTokenT<ElectronCollection> Electron_token;
  edm::Handle<ElectronCollection> electrons_;
  edm::Handle<TauCollection> taus_;

  PositionAtECalEntranceComputer positionAtECalEntrance_;

  static constexpr float ecalBarrelEndcapEtaBorder_ = 1.479;
  static constexpr float ecalEndcapVFEndcapEtaBorder_ = 2.4;

  bool vetoEcalCracks_;

  bool isPhase2_;
  float deltaREleTauMax_;

  int verbosity_;
};

template <class TauType, class TauDiscriminator, class ElectronType>
reco::SingleTauDiscriminatorContainer
TauDiscriminationAgainstElectronMVA6<TauType, TauDiscriminator, ElectronType>::discriminate(
    const TauRef& theTauRef) const {
  reco::SingleTauDiscriminatorContainer result;
  result.rawValues = {1., -1.};
  double category = -1.;
  bool isGsfElectronMatched = false;

  double deltaRDummy = 9.9;

  std::pair<float, float> tauEtaAtECalEntrance;
  if (std::is_same<TauType, reco::PFTau>::value || std::is_same<TauType, pat::Tau>::value)
    tauEtaAtECalEntrance = getTauEtaAtECalEntrance(theTauRef);
  else
    throw cms::Exception("TauDiscriminationAgainstElectronMVA6")
        << "Unsupported TauType used. You must use either reco::PFTau or pat::Tau.";

  if ((*theTauRef).leadChargedHadrCand().isNonnull()) {
    int numSignalGammaCandsInSigCone = 0;
    double signalrad = std::clamp(3.0 / std::max(1.0, theTauRef->pt()), 0.05, 0.10);
    for (const auto& gamma : theTauRef->signalGammaCands()) {
      double dR = deltaR(gamma->p4(), theTauRef->leadChargedHadrCand()->p4());
      // pfGammas inside the tau signal cone
      if (dR < signalrad) {
        numSignalGammaCandsInSigCone += 1;
      }
    }

    bool hasGsfTrack = false;
    const reco::CandidatePtr& leadChCand = theTauRef->leadChargedHadrCand();
    if (leadChCand.isNonnull()) {
      if (isPhase2_) {
        //MB: for phase-2 has gsf-track reads lead charged cand is pf-electron
        hasGsfTrack = (std::abs(leadChCand->pdgId()) == 11);
      } else {
        const pat::PackedCandidate* packedLeadChCand = dynamic_cast<const pat::PackedCandidate*>(leadChCand.get());
        if (packedLeadChCand != nullptr) {
          hasGsfTrack = (std::abs(packedLeadChCand->pdgId()) == 11);
        } else {
          const reco::PFCandidate* pfLeadChCand = dynamic_cast<const reco::PFCandidate*>(leadChCand.get());
          //pfLeadChCand can not be a nullptr here as it would be imply taus not built either with PFCandidates or PackedCandidates
          hasGsfTrack = pfLeadChCand->gsfTrackRef().isNonnull();
        }
      }
    }

    // loop over the electrons
    size_t iElec = 0;
    for (const auto& theElectron : *electrons_) {
      edm::Ref<ElectronCollection> theElecRef(electrons_, iElec);
      iElec++;
      if (theElectron.pt() > 10.) {  // CV: only take electrons above some minimal energy/Pt into account...
        double deltaREleTau = deltaR(theElectron.p4(), theTauRef->p4());
        deltaRDummy = std::min(deltaREleTau, deltaRDummy);
        if (deltaREleTau < deltaREleTauMax_) {
          double mva_match = mva_->mvaValue(*theTauRef, theElecRef);
          if (!hasGsfTrack)
            hasGsfTrack = theElectron.gsfTrack().isNonnull();

          // veto taus that go to ECal crack
          if (vetoEcalCracks_ &&
              (isInEcalCrack(tauEtaAtECalEntrance.first) || isInEcalCrack(tauEtaAtECalEntrance.second))) {
            // add category index
            result.rawValues.at(1) = category;
            // return MVA output value
            result.rawValues.at(0) = -99.;
            return result;
          }
          // veto taus that go to ECal crack

          if (std::abs(tauEtaAtECalEntrance.first) < ecalBarrelEndcapEtaBorder_) {  // Barrel
            if (numSignalGammaCandsInSigCone == 0 && hasGsfTrack) {
              category = 5.;
            } else if (numSignalGammaCandsInSigCone >= 1 && hasGsfTrack) {
              category = 7.;
            }
          } else if (!isPhase2_ || std::abs(tauEtaAtECalEntrance.first) < ecalEndcapVFEndcapEtaBorder_) {  // Endcap
            if (numSignalGammaCandsInSigCone == 0 && hasGsfTrack) {
              category = 13.;
            } else if (numSignalGammaCandsInSigCone >= 1 && hasGsfTrack) {
              category = 15.;
            }
          } else {  // VeryForwardEndcap
            if (numSignalGammaCandsInSigCone == 0 && hasGsfTrack) {
              category = 14.;
            } else if (numSignalGammaCandsInSigCone >= 1 && hasGsfTrack) {
              category = 16.;
            }
          }

          result.rawValues.at(0) = std::min(result.rawValues.at(0), float(mva_match));
          isGsfElectronMatched = true;
        }  // deltaR < deltaREleTauMax_
      }    // electron pt > 10
    }      // end of loop over electrons

    if (!isGsfElectronMatched) {
      double mva_nomatch = mva_->mvaValue(*theTauRef);

      // veto taus that go to ECal crack
      if (vetoEcalCracks_ &&
          (isInEcalCrack(tauEtaAtECalEntrance.first) || isInEcalCrack(tauEtaAtECalEntrance.second))) {
        // add category index
        result.rawValues.at(1) = category;
        // return MVA output value
        result.rawValues.at(0) = -99.;
        return result;
      }
      // veto taus that go to ECal crack

      if (std::abs(tauEtaAtECalEntrance.first) < ecalBarrelEndcapEtaBorder_) {  // Barrel
        if (numSignalGammaCandsInSigCone == 0 && !hasGsfTrack) {
          category = 0.;
        } else if (numSignalGammaCandsInSigCone >= 1 && !hasGsfTrack) {
          category = 2.;
        }
      } else if (!isPhase2_ || std::abs(tauEtaAtECalEntrance.first) < ecalEndcapVFEndcapEtaBorder_) {  // Endcap
        if (numSignalGammaCandsInSigCone == 0 && !hasGsfTrack) {
          category = 8.;
        } else if (numSignalGammaCandsInSigCone >= 1 && !hasGsfTrack) {
          category = 10.;
        }
      } else {  // VeryForwardEndcap
        if (numSignalGammaCandsInSigCone == 0 && !hasGsfTrack) {
          category = 9.;
        } else if (numSignalGammaCandsInSigCone >= 1 && !hasGsfTrack) {
          category = 11.;
        }
      }

      result.rawValues.at(0) = std::min(result.rawValues.at(0), float(mva_nomatch));
    }
  }

  if (verbosity_) {
    edm::LogPrint(this->getTauTypeString() + "AgainstEleMVA6")
        << "<" + this->getTauTypeString() + "AgainstElectronMVA6::discriminate>:";
    edm::LogPrint(this->getTauTypeString() + "AgainstEleMVA6")
        << " tau: Pt = " << theTauRef->pt() << ", eta = " << theTauRef->eta() << ", phi = " << theTauRef->phi();
    edm::LogPrint(this->getTauTypeString() + "AgainstEleMVA6")
        << " deltaREleTau = " << deltaRDummy << ", isGsfElectronMatched = " << isGsfElectronMatched;
    edm::LogPrint(this->getTauTypeString() + "AgainstEleMVA6")
        << " #Prongs = " << theTauRef->signalChargedHadrCands().size();
    edm::LogPrint(this->getTauTypeString() + "AgainstEleMVA6")
        << " MVA = " << result.rawValues.at(0) << ", category = " << category;
  }

  // add category index
  result.rawValues.at(1) = category;
  // return MVA output value
  return result;
}

template <class TauType, class TauDiscriminator, class ElectronType>
bool TauDiscriminationAgainstElectronMVA6<TauType, TauDiscriminator, ElectronType>::isInEcalCrack(double eta) const {
  double absEta = std::abs(eta);
  return (absEta > 1.460 && absEta < 1.558);
}

template <class TauType, class TauDiscriminator, class ElectronType>
std::pair<float, float>
TauDiscriminationAgainstElectronMVA6<TauType, TauDiscriminator, ElectronType>::getTauEtaAtECalEntrance(
    const reco::PFTauRef& theTauRef) const {
  float tauEtaAtECalEntrance = -99;
  float leadChargedCandEtaAtECalEntrance = -99;
  float sumEtaTimesEnergy = 0;
  float sumEnergy = 0;
  float leadChargedCandPt = -99;

  for (const auto& candidate : theTauRef->signalCands()) {
    float etaAtECalEntrance = candidate->eta();
    const reco::Track* track = nullptr;
    const reco::PFCandidate* pfCandidate = dynamic_cast<const reco::PFCandidate*>(candidate.get());
    if (pfCandidate != nullptr) {
      if (!isPhase2_ || std::abs(theTauRef->eta()) < ecalBarrelEndcapEtaBorder_) {  // ECal
        etaAtECalEntrance = pfCandidate->positionAtECALEntrance().eta();
      } else {  // HGCal
        bool success = false;
        reco::Candidate::Point posAtECal = positionAtECalEntrance_(candidate.get(), success);
        if (success) {
          etaAtECalEntrance = posAtECal.eta();
        }
      }
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
    } else {
      bool success = false;
      reco::Candidate::Point posAtECal = positionAtECalEntrance_(candidate.get(), success);
      if (success) {
        etaAtECalEntrance = posAtECal.eta();
      }
      track = candidate->bestTrack();
    }
    if (track != nullptr) {
      if (track->pt() > leadChargedCandPt) {
        leadChargedCandEtaAtECalEntrance = etaAtECalEntrance;
        leadChargedCandPt = track->pt();
      }
    }
    sumEtaTimesEnergy += etaAtECalEntrance * candidate->energy();
    sumEnergy += candidate->energy();
  }
  if (sumEnergy > 0.) {
    tauEtaAtECalEntrance = sumEtaTimesEnergy / sumEnergy;
  }
  return std::pair<float, float>(tauEtaAtECalEntrance, leadChargedCandEtaAtECalEntrance);
}

template <class TauType, class TauDiscriminator, class ElectronType>
std::pair<float, float>
TauDiscriminationAgainstElectronMVA6<TauType, TauDiscriminator, ElectronType>::getTauEtaAtECalEntrance(
    const pat::TauRef& theTauRef) const {
  if (!isPhase2_ || std::abs(theTauRef->eta()) < ecalBarrelEndcapEtaBorder_) {  // ECal
    return std::pair<float, float>(theTauRef->etaAtEcalEntrance(), theTauRef->etaAtEcalEntranceLeadChargedCand());
  } else {  // HGCal
    float tauEtaAtECalEntrance = -99;
    float leadChargedCandEtaAtECalEntrance = -99;
    float sumEtaTimesEnergy = 0.;
    float sumEnergy = 0.;
    float leadChargedCandPt = -99;

    for (const auto& candidate : theTauRef->signalCands()) {
      float etaAtECalEntrance = candidate->eta();
      bool success = false;
      reco::Candidate::Point posAtECal = positionAtECalEntrance_(candidate.get(), success);
      if (success) {
        etaAtECalEntrance = posAtECal.eta();
      }
      const reco::Track* track = candidate->bestTrack();
      if (track != nullptr) {
        if (track->pt() > leadChargedCandPt) {
          leadChargedCandEtaAtECalEntrance = etaAtECalEntrance;
          leadChargedCandPt = track->pt();
        }
      }
      sumEtaTimesEnergy += etaAtECalEntrance * candidate->energy();
      sumEnergy += candidate->energy();
    }
    if (sumEnergy > 0.) {
      tauEtaAtECalEntrance = sumEtaTimesEnergy / sumEnergy;
    }
    return std::pair<float, float>(tauEtaAtECalEntrance, leadChargedCandEtaAtECalEntrance);
  }
}

template <class TauType, class TauDiscriminator, class ElectronType>
void TauDiscriminationAgainstElectronMVA6<TauType, TauDiscriminator, ElectronType>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  // {pfReco,pat}TauDiscriminationAgainstElectronMVA6
  edm::ParameterSetDescription desc;

  desc.add<std::string>("method", "BDTG");
  desc.add<bool>("loadMVAfromDB", true);
  desc.add<bool>("returnMVA", true);

  desc.add<std::string>("mvaName_NoEleMatch_woGwoGSF_BL", "gbr_NoEleMatch_woGwoGSF_BL");
  desc.add<std::string>("mvaName_NoEleMatch_wGwoGSF_BL", "gbr_NoEleMatch_wGwoGSF_BL");
  desc.add<std::string>("mvaName_woGwGSF_BL", "gbr_woGwGSF_BL");
  desc.add<std::string>("mvaName_wGwGSF_BL", "gbr_wGwGSF_BL");
  desc.add<std::string>("mvaName_NoEleMatch_woGwoGSF_EC", "gbr_NoEleMatch_woGwoGSF_EC");
  desc.add<std::string>("mvaName_NoEleMatch_wGwoGSF_EC", "gbr_NoEleMatch_wGwoGSF_EC");
  desc.add<std::string>("mvaName_woGwGSF_EC", "gbr_woGwGSF_EC");
  desc.add<std::string>("mvaName_wGwGSF_EC", "gbr_wGwGSF_EC");

  desc.add<double>("minMVANoEleMatchWOgWOgsfBL", 0.0);
  desc.add<double>("minMVANoEleMatchWgWOgsfBL", 0.0);
  desc.add<double>("minMVAWOgWgsfBL", 0.0);
  desc.add<double>("minMVAWgWgsfBL", 0.0);
  desc.add<double>("minMVANoEleMatchWOgWOgsfEC", 0.0);
  desc.add<double>("minMVANoEleMatchWgWOgsfEC", 0.0);
  desc.add<double>("minMVAWOgWgsfEC", 0.0);
  desc.add<double>("minMVAWgWgsfEC", 0.0);

  desc.ifValue(
      edm::ParameterDescription<bool>("isPhase2", false, true),
      // MB: "srcElectrons" present for both phase-2 and non-phase2 to have a non-empy case for default, i.e. isPhase2=false
      false >> (edm::ParameterDescription<edm::InputTag>("srcElectrons", edm::InputTag("fixme"), true)) or
          // The following used only for Phase2
          true >> (edm::ParameterDescription<edm::InputTag>("srcElectrons", edm::InputTag("fixme"), true) and
                   edm::ParameterDescription<std::string>("mvaName_wGwGSF_VFEC", "gbr_wGwGSF_VFEC", true) and
                   edm::ParameterDescription<std::string>("mvaName_woGwGSF_VFEC", "gbr_woGwGSF_VFEC", true) and
                   edm::ParameterDescription<std::string>(
                       "mvaName_NoEleMatch_wGwoGSF_VFEC", "gbr_NoEleMatch_wGwoGSF_VFEC", true) and
                   edm::ParameterDescription<std::string>(
                       "mvaName_NoEleMatch_woGwoGSF_VFEC", "gbr_NoEleMatch_woGwoGSF_VFEC", true) and
                   edm::ParameterDescription<double>("minMVAWOgWgsfVFEC", 0.0, true) and
                   edm::ParameterDescription<double>("minMVAWgWgsfVFEC", 0.0, true) and
                   edm::ParameterDescription<double>("minMVANoEleMatchWgWOgsfVFEC", 0.0, true) and
                   edm::ParameterDescription<double>("minMVANoEleMatchWOgWOgsfVFEC", 0.0, true)));

  // Relevant only for gsfElectrons for Phase2
  if (std::is_same<ElectronType, reco::GsfElectron>::value) {
    desc.add<std::vector<edm::InputTag>>("hgcalElectronIDs", std::vector<edm::InputTag>())
        ->setComment("Relevant only for Phase-2");
  }
  desc.add<bool>("vetoEcalCracks", true);
  desc.add<bool>("usePhiAtEcalEntranceExtrapolation", false);
  desc.add<int>("verbosity", 0);

  TauDiscriminationProducerBase<TauType,
                                reco::TauDiscriminatorContainer,
                                reco::SingleTauDiscriminatorContainer,
                                TauDiscriminator>::fillProducerDescriptions(desc);  // inherited from the base-class

  descriptions.addWithDefaultLabel(desc);
}

typedef TauDiscriminationAgainstElectronMVA6<reco::PFTau, reco::PFTauDiscriminator, reco::GsfElectron>
    PFRecoTauDiscriminationAgainstElectronMVA6;
typedef TauDiscriminationAgainstElectronMVA6<pat::Tau, pat::PATTauDiscriminator, pat::Electron>
    PATTauDiscriminationAgainstElectronMVA6;

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA6);
DEFINE_FWK_MODULE(PATTauDiscriminationAgainstElectronMVA6);
