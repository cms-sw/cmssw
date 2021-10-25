/** \class ShiftedPFCandidateProducerForPFNoPUMEt
 *
 * Vary energy of PFCandidates which are (are not) within jets of Pt > 10 GeV
 * by jet energy uncertainty (by 10% "unclustered" energy uncertainty)
 *
 * NOTE: Auxiliary class specific to estimating systematic uncertainty
 *       on PFMET reconstructed by no-PU MET reconstruction algorithm
 *      (implemented in JetMETCorrections/Type1MET/src/PFNoPUMETProducer.cc)
 *
 *       In case all PFCandidates not within jets of Pt > 30 GeV would be varied
 *       by the 10% "unclustered" energy uncertainty, the systematic uncertainty
 *       on the reconstructed no-PU MET would be overestimated significantly !!
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include <string>
#include <vector>

class ShiftedPFCandidateProducerForPFNoPUMEt : public edm::stream::EDProducer<> {
public:
  explicit ShiftedPFCandidateProducerForPFNoPUMEt(const edm::ParameterSet&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::string moduleLabel_;

  edm::EDGetTokenT<reco::PFCandidateCollection> srcPFCandidatesToken_;
  edm::EDGetTokenT<reco::PFJetCollection> srcJetsToken_;

  edm::FileInPath jetCorrInputFileName_;
  std::string jetCorrPayloadName_;
  edm::ESGetToken<JetCorrectorParametersCollection, JetCorrectionsRecord> jetCorrPayloadToken_;
  std::string jetCorrUncertaintyTag_;
  std::unique_ptr<JetCorrectorParameters> jetCorrParameters_;
  std::unique_ptr<JetCorrectionUncertainty> jecUncertainty_;

  bool jecValidFileName_;

  double minJetPt_;

  double shiftBy_;

  double unclEnUncertainty_;
};

namespace {
  constexpr double dR2Match = 0.01 * 0.01;
}

ShiftedPFCandidateProducerForPFNoPUMEt::ShiftedPFCandidateProducerForPFNoPUMEt(const edm::ParameterSet& cfg)
    : srcPFCandidatesToken_(consumes<reco::PFCandidateCollection>(cfg.getParameter<edm::InputTag>("srcPFCandidates"))),
      srcJetsToken_(consumes<reco::PFJetCollection>(cfg.getParameter<edm::InputTag>("srcJets"))) {
  jetCorrUncertaintyTag_ = cfg.getParameter<std::string>("jetCorrUncertaintyTag");

  jecValidFileName_ = cfg.exists("jetCorrInputFileName");
  if (jecValidFileName_) {
    jetCorrInputFileName_ = cfg.getParameter<edm::FileInPath>("jetCorrInputFileName");
    if (jetCorrInputFileName_.location() == edm::FileInPath::Unknown)
      throw cms::Exception("ShiftedJetProducerT")
          << " Failed to find JEC parameter file = " << jetCorrInputFileName_ << " !!\n";
    edm::LogInfo("ShiftedPFCandidateProducerForPFNoPUMEt")
        << "Reading JEC parameters = " << jetCorrUncertaintyTag_ << " from file = " << jetCorrInputFileName_.fullPath()
        << "." << std::endl;
    jetCorrParameters_ =
        std::make_unique<JetCorrectorParameters>(jetCorrInputFileName_.fullPath(), jetCorrUncertaintyTag_);
    jecUncertainty_ = std::make_unique<JetCorrectionUncertainty>(*jetCorrParameters_);
  } else {
    edm::LogInfo("ShiftedPFCandidateProducerForPFNoPUMEt")
        << "Reading JEC parameters = " << jetCorrUncertaintyTag_ << " from DB/SQLlite file." << std::endl;
    jetCorrPayloadName_ = cfg.getParameter<std::string>("jetCorrPayloadName");
    jetCorrPayloadToken_ = esConsumes(edm::ESInputTag("", jetCorrPayloadName_));
  }

  minJetPt_ = cfg.getParameter<double>("minJetPt");

  shiftBy_ = cfg.getParameter<double>("shiftBy");

  unclEnUncertainty_ = cfg.getParameter<double>("unclEnUncertainty");

  produces<reco::PFCandidateCollection>();
}

void ShiftedPFCandidateProducerForPFNoPUMEt::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<reco::PFCandidateCollection> originalPFCandidates;
  evt.getByToken(srcPFCandidatesToken_, originalPFCandidates);

  edm::Handle<reco::PFJetCollection> jets;
  evt.getByToken(srcJetsToken_, jets);

  std::vector<const reco::PFJet*> selectedJets;
  for (reco::PFJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
    if (jet->pt() > minJetPt_)
      selectedJets.push_back(&(*jet));
  }

  if (!jetCorrPayloadName_.empty()) {
    JetCorrectorParametersCollection const& jetCorrParameterSet = es.getData(jetCorrPayloadToken_);
    const JetCorrectorParameters& jetCorrParameters = (jetCorrParameterSet)[jetCorrUncertaintyTag_];
    jecUncertainty_ = std::make_unique<JetCorrectionUncertainty>(jetCorrParameters);
  }

  auto shiftedPFCandidates = std::make_unique<reco::PFCandidateCollection>();

  for (reco::PFCandidateCollection::const_iterator originalPFCandidate = originalPFCandidates->begin();
       originalPFCandidate != originalPFCandidates->end();
       ++originalPFCandidate) {
    const reco::PFJet* jet_matched = nullptr;
    for (auto jet : selectedJets) {
      for (const auto& jetc : jet->getPFConstituents()) {
        if (deltaR2(originalPFCandidate->p4(), jetc->p4()) < dR2Match) {
          jet_matched = jet;
          break;
        }
      }
      if (jet_matched)
        break;
    }

    double shift = 0.;
    if (jet_matched != nullptr) {
      jecUncertainty_->setJetEta(jet_matched->eta());
      jecUncertainty_->setJetPt(jet_matched->pt());

      shift = jecUncertainty_->getUncertainty(true);
    } else {
      shift = unclEnUncertainty_;
    }

    shift *= shiftBy_;

    reco::Candidate::LorentzVector shiftedPFCandidateP4 = originalPFCandidate->p4();
    shiftedPFCandidateP4 *= (1. + shift);

    reco::PFCandidate shiftedPFCandidate(*originalPFCandidate);
    shiftedPFCandidate.setP4(shiftedPFCandidateP4);

    shiftedPFCandidates->push_back(shiftedPFCandidate);
  }

  evt.put(std::move(shiftedPFCandidates));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ShiftedPFCandidateProducerForPFNoPUMEt);
