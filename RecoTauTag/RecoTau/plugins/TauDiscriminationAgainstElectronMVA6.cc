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
class TauDiscriminationAgainstElectronMVA6 : public TauDiscriminationProducerBase<TauType, reco::TauDiscriminatorContainer, reco::SingleTauDiscriminatorContainer, TauDiscriminator> {
public:
  typedef std::vector<TauType> TauCollection;
  typedef edm::Ref<TauCollection> TauRef;
  typedef std::vector<ElectronType> ElectronCollection;

  explicit TauDiscriminationAgainstElectronMVA6(const edm::ParameterSet& cfg)
    : TauDiscriminationProducerBase<TauType, reco::TauDiscriminatorContainer, reco::SingleTauDiscriminatorContainer, TauDiscriminator>::TauDiscriminationProducerBase(cfg),
      moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      mva_(std::make_unique<AntiElectronIDMVA6<TauType, ElectronType> >(cfg)),
      Electron_token(edm::EDConsumerBase::consumes<ElectronCollection>(cfg.getParameter<edm::InputTag>("srcElectrons"))), //MB: full specification with prefix mandatory
      vetoEcalCracks_(cfg.getParameter<bool>("vetoEcalCracks")),
      verbosity_(cfg.getParameter<int>("verbosity")) {}

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
  std::pair<float,float> getTauEtaAtECalEntrance(const reco::PFTauRef& theTauRef) const;
  std::pair<float,float> getTauEtaAtECalEntrance(const pat::TauRef& theTauRef) const;

  std::string moduleLabel_;
  std::unique_ptr<AntiElectronIDMVA6<TauType, ElectronType> > mva_;

  edm::EDGetTokenT<ElectronCollection> Electron_token;
  edm::Handle<ElectronCollection> electrons_;
  edm::Handle<TauCollection> taus_;

  PositionAtECalEntranceComputer positionAtECalEntrance_;

  bool vetoEcalCracks_;

  int verbosity_;
};

template <class TauType, class TauDiscriminator, class ElectronType>
reco::SingleTauDiscriminatorContainer 
TauDiscriminationAgainstElectronMVA6<TauType, TauDiscriminator, ElectronType>::discriminate(const TauRef& theTauRef) const {

  reco::SingleTauDiscriminatorContainer result;
  result.rawValues = {1., -1.};
  double category = -1.;
  bool isGsfElectronMatched = false;

  float deltaRDummy = 9.9;

  const float ECalBarrelEndcapEtaBorder = 1.479;

  std::pair<float,float> tauEtaAtECalEntrance; 
  if (std::is_same<TauType, reco::PFTau>::value ||
      std::is_same<TauType, pat::Tau>::value)
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
      const pat::PackedCandidate* packedLeadChCand =
	dynamic_cast<const pat::PackedCandidate*>(leadChCand.get());
      if (packedLeadChCand != nullptr) {
	hasGsfTrack = (std::abs(packedLeadChCand->pdgId()) == 11);
      } else {
	const reco::PFCandidate* pfLeadChCand =
	  dynamic_cast<const reco::PFCandidate*>(leadChCand.get());
	hasGsfTrack = (pfLeadChCand != nullptr && pfLeadChCand->gsfTrackRef().isNonnull());
      }
    }

    // loop over the electrons
    for (const auto& theElectron : *electrons_) {
      if (theElectron.pt() > 10.) {  // CV: only take electrons above some minimal energy/Pt into account...
        double deltaREleTau = deltaR(theElectron.p4(), theTauRef->p4());
        deltaRDummy = deltaREleTau;
        if (deltaREleTau < 0.3) {
          double mva_match = mva_->MVAValue(*theTauRef, theElectron);
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

          if (std::abs(tauEtaAtECalEntrance.first) < ECalBarrelEndcapEtaBorder) {  // Barrel
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
      double mva_nomatch = mva_->MVAValue(*theTauRef);

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

      if (std::abs(tauEtaAtECalEntrance.first) < ECalBarrelEndcapEtaBorder) {  // Barrel
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

      result.rawValues.at(0) = std::min(result.rawValues.at(0), float(mva_nomatch));
    }
  }

  if (verbosity_) {
    edm::LogPrint(this->getTauTypeString() + "AgainstEleMVA6")
      << "<" + this->getTauTypeString() + "AgainstElectronMVA6::discriminate>:";
    edm::LogPrint(this->getTauTypeString() + "AgainstEleMVA6")
      << " tau: Pt = " << theTauRef->pt() << ", eta = " << theTauRef->eta()
      << ", phi = " << theTauRef->phi();
    edm::LogPrint(this->getTauTypeString() + "AgainstEleMVA6")
      << " deltaREleTau = " << deltaRDummy
      << ", isGsfElectronMatched = " << isGsfElectronMatched;
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
bool TauDiscriminationAgainstElectronMVA6<TauType, TauDiscriminator, ElectronType>::isInEcalCrack(
double eta) const {
  double absEta = std::abs(eta);
  return (absEta > 1.460 && absEta < 1.558);
}

template <class TauType, class TauDiscriminator, class ElectronType>
std::pair<float,float> TauDiscriminationAgainstElectronMVA6<TauType, TauDiscriminator, ElectronType>::getTauEtaAtECalEntrance(const reco::PFTauRef& theTauRef) const {

  float tauEtaAtECalEntrance = -99;
  float leadChargedCandEtaAtECalEntrance = -99;
  float sumEtaTimesEnergy = 0;
  float sumEnergy = 0;
  float leadChargedCandPt = -99;

  for (const auto& candidate : theTauRef->signalCands()) {
    float etaAtECalEntrance = candidate->eta();
    const reco::Track* track = nullptr;
    const reco::PFCandidate* pfCandidate =
      dynamic_cast<const reco::PFCandidate*>(candidate.get());    
    if (pfCandidate != nullptr) {
      etaAtECalEntrance = pfCandidate->positionAtECALEntrance().eta();      
      if (pfCandidate->trackRef().isNonnull())
	track = pfCandidate->trackRef().get();
      else if (pfCandidate->muonRef().isNonnull() &&
	       pfCandidate->muonRef()->innerTrack().isNonnull())
	track = pfCandidate->muonRef()->innerTrack().get();
      else if (pfCandidate->muonRef().isNonnull() &&
	       pfCandidate->muonRef()->globalTrack().isNonnull())
	track = pfCandidate->muonRef()->globalTrack().get();
      else if (pfCandidate->muonRef().isNonnull() &&
	       pfCandidate->muonRef()->outerTrack().isNonnull())
	track = pfCandidate->muonRef()->outerTrack().get();
      else if (pfCandidate->gsfTrackRef().isNonnull())
	track = pfCandidate->gsfTrackRef().get();
    } else {
      bool success = false;
      reco::Candidate::Point posAtECal = 
	positionAtECalEntrance_(candidate.get(), success);
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
  return std::pair<float,float>(tauEtaAtECalEntrance,
				leadChargedCandEtaAtECalEntrance);
}

template <class TauType, class TauDiscriminator, class ElectronType>
std::pair<float,float> TauDiscriminationAgainstElectronMVA6<TauType, TauDiscriminator, ElectronType>::getTauEtaAtECalEntrance(const pat::TauRef& theTauRef) const {
  return std::pair<float,float>(theTauRef->etaAtEcalEntrance(),
				theTauRef->etaAtEcalEntranceLeadChargedCand());
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
  desc.add<edm::InputTag>("srcElectrons", edm::InputTag("fixme"));
  desc.add<bool>("vetoEcalCracks", true);
  desc.add<bool>("usePhiAtEcalEntranceExtrapolation", false);
  desc.add<int>("verbosity", 0);
  
  TauDiscriminationProducerBase<TauType, reco::TauDiscriminatorContainer, reco::SingleTauDiscriminatorContainer, TauDiscriminator>::fillProducerDescriptions(desc); // inherited from the base-class

  descriptions.addWithDefaultLabel(desc);
}

typedef TauDiscriminationAgainstElectronMVA6<reco::PFTau, reco::PFTauDiscriminator, reco::GsfElectron>
    PFRecoTauDiscriminationAgainstElectronMVA6;
typedef TauDiscriminationAgainstElectronMVA6<pat::Tau, pat::PATTauDiscriminator, pat::Electron>
    PATTauDiscriminationAgainstElectronMVA6;

DEFINE_FWK_MODULE(PFRecoTauDiscriminationAgainstElectronMVA6);
DEFINE_FWK_MODULE(PATTauDiscriminationAgainstElectronMVA6);
