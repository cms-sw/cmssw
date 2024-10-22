#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <iostream>

class GEDValueMapAnalyzer : public edm::one::EDAnalyzer<> {
public:
  GEDValueMapAnalyzer(const edm::ParameterSet&);
  ~GEDValueMapAnalyzer() override = default;

  void analyze(const edm::Event& iEvent, const edm::EventSetup& c) override;

private:
  const edm::InputTag inputTagPFCandidates_;
  const edm::InputTag inputTagValueMapElectrons_;
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfCandToken_;
  const edm::EDGetTokenT<edm::ValueMap<reco::GsfElectronRef> > electronToken_;
};

GEDValueMapAnalyzer::GEDValueMapAnalyzer(const edm::ParameterSet& iConfig)
    : inputTagPFCandidates_(iConfig.getParameter<edm::InputTag>("PFCandidates")),
      inputTagValueMapElectrons_(iConfig.getParameter<edm::InputTag>("ElectronValueMap")),
      pfCandToken_(consumes<reco::PFCandidateCollection>(inputTagPFCandidates_)),
      electronToken_(consumes<edm::ValueMap<reco::GsfElectronRef> >(inputTagValueMapElectrons_)) {}

void GEDValueMapAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& c) {
  const edm::Handle<reco::PFCandidateCollection>& pfCandidatesH = iEvent.getHandle(pfCandToken_);
  if (!pfCandidatesH.isValid()) {
    std::ostringstream err;
    err << " cannot get PFCandidates: " << inputTagPFCandidates_ << std::endl;
    edm::LogError("PFIsoReader") << err.str();
    throw cms::Exception("MissingProduct", err.str());
  }

  // Get the value maps

  const edm::Handle<edm::ValueMap<reco::GsfElectronRef> > electronValMapH = iEvent.getHandle(electronToken_);
  const edm::ValueMap<reco::GsfElectronRef>& myElectronValMap(*electronValMapH);

  edm::LogVerbatim("GEDValueMapAnalyer") << " Read Electron Value Map " << myElectronValMap.size();

  unsigned ncandidates = pfCandidatesH->size();
  for (unsigned ic = 0; ic < ncandidates; ++ic) {
    // check if it has a GsfTrack
    const reco::PFCandidate& cand((*pfCandidatesH)[ic]);
    if (!cand.gsfTrackRef().isNonnull())
      continue;

    reco::PFCandidateRef pfRef(pfCandidatesH, ic);
    // get the GsfElectronRef from the ValueMap
    reco::GsfElectronRef gsfRef = myElectronValMap[pfRef];

    //basic check
    std::ostringstream st1;
    st1 << " Comparing GsfTrackRef from GsfElectron and PFCandidate ";
    if (gsfRef->gsfTrack() == cand.gsfTrackRef())
      st1 << " OK ";
    else
      st1 << " Problem different Ref";
    edm::LogVerbatim("GEDValueMapAnalyer") << st1.str();
  }
}

DEFINE_FWK_MODULE(GEDValueMapAnalyzer);
