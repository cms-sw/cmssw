/** \class ShiftedPFCandidateProducerByMatchedObject
 *
 * Vary energy of PFCandidates coinciding in eta-phi with selected electrons/muons/tau-jets/jets
 * by electron/muon/tau-jet/jet energy uncertainty.
 *
 * \author Christian Veelken, LLR
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <string>
#include <vector>

class ShiftedPFCandidateProducerByMatchedObject : public edm::stream::EDProducer<> {
public:
  explicit ShiftedPFCandidateProducerByMatchedObject(const edm::ParameterSet&);
  ~ShiftedPFCandidateProducerByMatchedObject() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidates_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > srcUnshiftedObjects_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > srcShiftedObjects_;

  double dRmatch_PFCandidate_;
  double dR2match_PFCandidate_;
  double dRmatch_Object_;
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
    double dRmatch_;
    double shift_;
    bool isValidMatch_;
  };

  std::vector<objectEntryType> objects_;
};

const double dRDefault = 1000;

ShiftedPFCandidateProducerByMatchedObject::ShiftedPFCandidateProducerByMatchedObject(const edm::ParameterSet& cfg) {
  srcPFCandidates_ = consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("srcPFCandidates"));
  srcUnshiftedObjects_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("srcUnshiftedObjects"));
  srcShiftedObjects_ = consumes<edm::View<reco::Candidate> >(cfg.getParameter<edm::InputTag>("srcShiftedObjects"));

  dRmatch_PFCandidate_ = cfg.getParameter<double>("dRmatch_PFCandidate");
  dR2match_PFCandidate_ = dRmatch_PFCandidate_ * dRmatch_PFCandidate_;
  dRmatch_Object_ = cfg.exists("dRmatch_Object") ? cfg.getParameter<double>("dRmatch_Object") : 0.1;
  dR2match_Object_ = dRmatch_Object_ * dRmatch_Object_;
  produces<reco::PFCandidateCollection>();
}

ShiftedPFCandidateProducerByMatchedObject::~ShiftedPFCandidateProducerByMatchedObject() {
  // nothing to be done yet...
}

void ShiftedPFCandidateProducerByMatchedObject::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<reco::PFCandidateCollection> originalPFCandidates;
  evt.getByToken(srcPFCandidates_, originalPFCandidates);

  typedef edm::View<reco::Candidate> CandidateView;

  edm::Handle<CandidateView> unshiftedObjects;
  evt.getByToken(srcUnshiftedObjects_, unshiftedObjects);

  edm::Handle<CandidateView> shiftedObjects;
  evt.getByToken(srcShiftedObjects_, shiftedObjects);

  objects_.clear();

  CandidateView::const_iterator shiftedObjectP4_matched;
  bool isMatched_Object = false;
  double dR2bestMatch_Object = dRDefault;
  for (CandidateView::const_iterator unshiftedObject = unshiftedObjects->begin();
       unshiftedObject != unshiftedObjects->end();
       ++unshiftedObject) {
    isMatched_Object = false;
    dR2bestMatch_Object = dRDefault;

    for (CandidateView::const_iterator shiftedObject = shiftedObjects->begin(); shiftedObject != shiftedObjects->end();
         ++shiftedObject) {
      double dR2 = deltaR2(unshiftedObject->p4(), shiftedObject->p4());
      if (dR2 < dR2match_Object_ && dR2 < dR2bestMatch_Object) {
        shiftedObjectP4_matched = shiftedObject;
        isMatched_Object = true;
        dR2bestMatch_Object = dR2;
      }
    }
    if (isMatched_Object) {
      objects_.push_back(
          objectEntryType(shiftedObjectP4_matched->p4(), unshiftedObject->p4(), sqrt(dR2bestMatch_Object)));
    }
  }

  auto shiftedPFCandidates = std::make_unique<reco::PFCandidateCollection>();

  for (reco::PFCandidateCollection::const_iterator originalPFCandidate = originalPFCandidates->begin();
       originalPFCandidate != originalPFCandidates->end();
       ++originalPFCandidate) {
    double shift = 0.;
    bool applyShift = false;
    double dR2bestMatch_PFCandidate = dRDefault;
    for (std::vector<objectEntryType>::const_iterator object = objects_.begin(); object != objects_.end(); ++object) {
      if (!object->isValidMatch_)
        continue;
      double dR2 = deltaR2(originalPFCandidate->p4(), object->unshiftedObjectP4_);
      if (dR2 < dR2match_PFCandidate_ && dR2 < dR2bestMatch_PFCandidate) {
        shift = object->shift_;
        applyShift = true;
        dR2bestMatch_PFCandidate = dR2;
      }
    }

    reco::Candidate::LorentzVector shiftedPFCandidateP4 = originalPFCandidate->p4();
    if (applyShift) {
      double shiftedPx = (1. + shift) * originalPFCandidate->px();
      double shiftedPy = (1. + shift) * originalPFCandidate->py();
      double shiftedPz = (1. + shift) * originalPFCandidate->pz();
      double mass = originalPFCandidate->mass();
      double shiftedEn = sqrt(shiftedPx * shiftedPx + shiftedPy * shiftedPy + shiftedPz * shiftedPz + mass * mass);
      shiftedPFCandidateP4.SetPxPyPzE(shiftedPx, shiftedPy, shiftedPz, shiftedEn);
    }

    reco::PFCandidate shiftedPFCandidate(*originalPFCandidate);
    shiftedPFCandidate.setP4(shiftedPFCandidateP4);

    shiftedPFCandidates->push_back(shiftedPFCandidate);
  }

  evt.put(std::move(shiftedPFCandidates));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedPFCandidateProducerByMatchedObject);
