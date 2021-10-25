/** \class ShiftedPFCandidateProducerForPFMVAMEt
 *
 * Vary energy of PFCandidates coinciding in eta-phi with selected electrons/muons/tau-jets/jets
 * by electron/muon/tau-jet/jet energy uncertainty.
 *
 * NOTE: Auxiliary class specific to estimating systematic uncertainty
 *       on PFMET reconstructed by MVA-based algorithm
 *      (implemented in RecoMET/METPUSubtraction/src/PFMETProducerMVA.cc)
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"
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

class ShiftedPFCandidateProducerForPFMVAMEt : public edm::stream::EDProducer<> {
public:
  explicit ShiftedPFCandidateProducerForPFMVAMEt(const edm::ParameterSet&);
  ~ShiftedPFCandidateProducerForPFMVAMEt() override;

private:
  typedef edm::View<reco::Candidate> CandidateView;

  void produce(edm::Event&, const edm::EventSetup&) override;

  std::string moduleLabel_;

  edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidatesToken_;
  edm::EDGetTokenT<CandidateView> srcUnshiftedObjectsToken_;
  edm::EDGetTokenT<CandidateView> srcShiftedObjectsToken_;

  double dRmatch_PFCandidate_;
  double dRmatch_Object_;

  double dR2match_PFCandidate_;
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

ShiftedPFCandidateProducerForPFMVAMEt::ShiftedPFCandidateProducerForPFMVAMEt(const edm::ParameterSet& cfg)
    : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
      srcPFCandidatesToken_(consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("srcPFCandidates"))),
      srcUnshiftedObjectsToken_(consumes<CandidateView>(cfg.getParameter<edm::InputTag>("srcUnshiftedObjects"))),
      srcShiftedObjectsToken_(consumes<CandidateView>(cfg.getParameter<edm::InputTag>("srcShiftedObjects"))) {
  dRmatch_PFCandidate_ = cfg.getParameter<double>("dRmatch_PFCandidate");
  dRmatch_Object_ = cfg.exists("dRmatch_Object") ? cfg.getParameter<double>("dRmatch_Object") : 0.1;

  dR2match_PFCandidate_ = dRmatch_PFCandidate_ * dRmatch_PFCandidate_;
  dR2match_Object_ = dRmatch_Object_ * dRmatch_Object_;

  produces<reco::PFCandidateCollection>();
}

ShiftedPFCandidateProducerForPFMVAMEt::~ShiftedPFCandidateProducerForPFMVAMEt() {
  // nothing to be done yet...
}

void ShiftedPFCandidateProducerForPFMVAMEt::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<reco::PFCandidateCollection> originalPFCandidates;
  evt.getByToken(srcPFCandidatesToken_, originalPFCandidates);

  edm::Handle<CandidateView> unshiftedObjects;
  evt.getByToken(srcUnshiftedObjectsToken_, unshiftedObjects);

  edm::Handle<CandidateView> shiftedObjects;
  evt.getByToken(srcShiftedObjectsToken_, shiftedObjects);

  objects_.clear();

  std::vector<bool> match(shiftedObjects->size(), false);
  int prevMatch = -1;
  int cnt = 0;

  for (CandidateView::const_iterator unshiftedObject = unshiftedObjects->begin();
       unshiftedObject != unshiftedObjects->end();
       ++unshiftedObject) {
    bool isMatched_Object = false;
    double dR2bestMatch_Object = dRDefault;
    reco::Candidate::LorentzVector shiftedObjectP4_matched;
    prevMatch = -1;

    for (CandidateView::const_iterator shiftedObject = shiftedObjects->begin(); shiftedObject != shiftedObjects->end();
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

  auto shiftedPFCandidates = std::make_unique<reco::PFCandidateCollection>();

  for (reco::PFCandidateCollection::const_iterator originalPFCandidate = originalPFCandidates->begin();
       originalPFCandidate != originalPFCandidates->end();
       ++originalPFCandidate) {
    double shift = 0.;
    bool applyShift = false;
    double dR2bestMatch_PFCandidate = dRDefault;
    prevMatch = -1;
    cnt = 0;

    for (std::vector<objectEntryType>::const_iterator object = objects_.begin(); object != objects_.end(); ++object) {
      if (!object->isValidMatch_)
        continue;
      if (match[cnt])
        continue;

      double dR2 = deltaR2(originalPFCandidate->p4(), object->unshiftedObjectP4_);
      if (dR2 < dR2match_PFCandidate_ && dR2 < dR2bestMatch_PFCandidate) {
        shift = object->shift_;
        applyShift = true;
        dR2bestMatch_PFCandidate = dR2;

        prevMatch = cnt;
      }
      cnt++;
    }

    reco::Candidate::LorentzVector shiftedPFCandidateP4 = originalPFCandidate->p4();
    if (applyShift) {
      //Ambiguity removal
      match[prevMatch] = true;

      shiftedPFCandidateP4 *= (1. + shift);
    }

    reco::PFCandidate shiftedPFCandidate(*originalPFCandidate);
    shiftedPFCandidate.setP4(shiftedPFCandidateP4);

    shiftedPFCandidates->push_back(shiftedPFCandidate);
  }

  evt.put(std::move(shiftedPFCandidates));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedPFCandidateProducerForPFMVAMEt);
