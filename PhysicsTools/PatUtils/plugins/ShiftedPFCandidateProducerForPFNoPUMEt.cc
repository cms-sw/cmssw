#include "PhysicsTools/PatUtils/plugins/ShiftedPFCandidateProducerForPFNoPUMEt.h"

#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Math/interface/deltaR.h"

const double dR2Match = 0.01 * 0.01;

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
    jetCorrParameters_ = new JetCorrectorParameters(jetCorrInputFileName_.fullPath(), jetCorrUncertaintyTag_);
    jecUncertainty_ = new JetCorrectionUncertainty(*jetCorrParameters_);
  } else {
    edm::LogInfo("ShiftedPFCandidateProducerForPFNoPUMEt")
        << "Reading JEC parameters = " << jetCorrUncertaintyTag_ << " from DB/SQLlite file." << std::endl;
    jetCorrPayloadName_ = cfg.getParameter<std::string>("jetCorrPayloadName");
  }

  minJetPt_ = cfg.getParameter<double>("minJetPt");

  shiftBy_ = cfg.getParameter<double>("shiftBy");

  unclEnUncertainty_ = cfg.getParameter<double>("unclEnUncertainty");

  produces<reco::PFCandidateCollection>();
}

ShiftedPFCandidateProducerForPFNoPUMEt::~ShiftedPFCandidateProducerForPFNoPUMEt() {
  if (jecValidFileName_) {
    delete jetCorrParameters_;
    delete jecUncertainty_;
  }
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
    edm::ESHandle<JetCorrectorParametersCollection> jetCorrParameterSet;
    es.get<JetCorrectionsRecord>().get(jetCorrPayloadName_, jetCorrParameterSet);
    const JetCorrectorParameters& jetCorrParameters = (*jetCorrParameterSet)[jetCorrUncertaintyTag_];
    delete jecUncertainty_;
    jecUncertainty_ = new JetCorrectionUncertainty(jetCorrParameters);
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
