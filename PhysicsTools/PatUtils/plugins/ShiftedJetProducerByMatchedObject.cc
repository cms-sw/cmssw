/** \class ShiftedJetProducerByMatchedObject
 *
 * Vary energy of jets coinciding in eta-phi with selected electrons/muons/tau-jets
 * by electron/muon/tau-jet energy uncertainty.
 *
 * \author Christian Veelken, LLR
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>

template <typename T>
class ShiftedJetProducerByMatchedObjectT : public edm::stream::EDProducer<> {
  typedef std::vector<T> JetCollection;

public:
  explicit ShiftedJetProducerByMatchedObjectT(const edm::ParameterSet&);
  ~ShiftedJetProducerByMatchedObjectT() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::string moduleLabel_;

  edm::EDGetTokenT<JetCollection> srcJets_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > srcUnshiftedObjects_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > srcShiftedObjects_;

  double dRmatch_Jet_;
  double dRmatch_Object_;

  double dR2match_Jet_;
  double dR2match_Object_;

  struct objectEntryType {
    objectEntryType(const reco::Candidate::LorentzVector& shiftedObjectP4,
                    const reco::Candidate::LorentzVector& unshiftedObjectP4,
                    double dRmatch)
        : shiftedObjectP4_(shiftedObjectP4),
          unshiftedObjectP4_(unshiftedObjectP4),
          dRmatch_(dRmatch),
          isValidMatch_(false) {
      if (unshiftedObjectP4.energy() > 0.) {
        shift_ = (shiftedObjectP4.energy() / unshiftedObjectP4.energy()) - 1.;
        isValidMatch_ = true;
      }
    }
    ~objectEntryType() {}
    reco::Candidate::LorentzVector shiftedObjectP4_;
    reco::Candidate::LorentzVector unshiftedObjectP4_;
    double shift_;
    double dRmatch_;
    bool isValidMatch_;
  };

  std::vector<objectEntryType> objects_;
};

template <typename T>
ShiftedJetProducerByMatchedObjectT<T>::ShiftedJetProducerByMatchedObjectT(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")) {
  srcJets_ = consumes<JetCollection>(cfg.getParameter<edm::InputTag>("srcJets"));
  srcUnshiftedObjects_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("srcUnshiftedObjects"));
  srcShiftedObjects_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("srcShiftedObjects"));

  dRmatch_Jet_ = cfg.getParameter<double>("dRmatch_Jet");
  dRmatch_Object_ = cfg.exists("dRmatch_Object") ? cfg.getParameter<double>("dRmatch_Object") : 0.1;

  dR2match_Jet_ = dRmatch_Jet_ * dRmatch_Jet_;
  dR2match_Object_ = dRmatch_Object_ * dRmatch_Object_;

  produces<JetCollection>();
}

template <typename T>
ShiftedJetProducerByMatchedObjectT<T>::~ShiftedJetProducerByMatchedObjectT() {
  // nothing to be done yet...
}

template <typename T>
void ShiftedJetProducerByMatchedObjectT<T>::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<JetCollection> originalJets;
  evt.getByToken(srcJets_, originalJets);

  edm::Handle<reco::CandidateView> unshiftedObjects;
  evt.getByToken(srcUnshiftedObjects_, unshiftedObjects);

  edm::Handle<reco::CandidateView> shiftedObjects;
  evt.getByToken(srcShiftedObjects_, shiftedObjects);

  objects_.clear();

  std::vector<bool> match(shiftedObjects->size(), false);
  int prevMatch = -1;
  int cnt = 0;

  for (reco::CandidateView::const_iterator unshiftedObject = unshiftedObjects->begin();
       unshiftedObject != unshiftedObjects->end();
       ++unshiftedObject) {
    bool isMatched_Object = false;
    double dR2bestMatch_Object = std::numeric_limits<double>::max();
    prevMatch = -1;
    cnt = 0;

    reco::Candidate::LorentzVector shiftedObjectP4_matched;
    for (reco::CandidateView::const_iterator shiftedObject = shiftedObjects->begin();
         shiftedObject != shiftedObjects->end();
         ++shiftedObject) {
      if (match[cnt])
        continue;

      double dR2 = deltaR2(unshiftedObject->p4(), shiftedObject->p4());
      if (dR2 < dR2match_Object_ && dR2 < dR2bestMatch_Object) {
        shiftedObjectP4_matched = shiftedObject->p4();
        isMatched_Object = true;
        dR2bestMatch_Object = dR2;

        prevMatch = cnt;
      }
      cnt++;
    }
    if (isMatched_Object) {
      //Ambiguity removal
      match[prevMatch] = true;
      objects_.push_back(objectEntryType(shiftedObjectP4_matched, unshiftedObject->p4(), sqrt(dR2bestMatch_Object)));
    }
  }

  match.assign(objects_.size(), false);

  auto shiftedJets = std::make_unique<JetCollection>();

  for (typename JetCollection::const_iterator originalJet = originalJets->begin(); originalJet != originalJets->end();
       ++originalJet) {
    double shift = 0.;
    bool applyShift = false;
    double dR2bestMatch_Jet = std::numeric_limits<double>::max();
    prevMatch = -1;
    cnt = 0;

    for (typename std::vector<objectEntryType>::const_iterator object = objects_.begin(); object != objects_.end();
         ++object) {
      if (!object->isValidMatch_)
        continue;
      if (match[cnt])
        continue;

      double dR2 = deltaR2(originalJet->p4(), object->unshiftedObjectP4_);
      if (dR2 < dR2match_Jet_ && dR2 < dR2bestMatch_Jet) {
        shift = object->shift_;
        applyShift = true;
        dR2bestMatch_Jet = dR2;

        prevMatch = cnt;
      }
      cnt++;
    }

    reco::Candidate::LorentzVector shiftedJetP4 = originalJet->p4();
    if (applyShift) {
      //Ambiguity removal
      match[prevMatch] = true;

      shiftedJetP4 *= (1. + shift);
    }

    T shiftedJet(*originalJet);
    shiftedJet.setP4(shiftedJetP4);

    shiftedJets->push_back(shiftedJet);
  }

  evt.put(std::move(shiftedJets));
}

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"

typedef ShiftedJetProducerByMatchedObjectT<reco::CaloJet> ShiftedCaloJetProducerByMatchedObject;
typedef ShiftedJetProducerByMatchedObjectT<reco::PFJet> ShiftedPFJetProducerByMatchedObject;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedCaloJetProducerByMatchedObject);
DEFINE_FWK_MODULE(ShiftedPFJetProducerByMatchedObject);
