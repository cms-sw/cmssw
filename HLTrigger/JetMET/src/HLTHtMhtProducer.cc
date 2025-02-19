
#include "HLTrigger/JetMET/interface/HLTHtMhtProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"


HLTHtMhtProducer::HLTHtMhtProducer(const edm::ParameterSet & iConfig) :
  usePt_          ( iConfig.getParameter<bool>("usePt") ),
  useTracks_      ( iConfig.getParameter<bool>("useTracks") ),
  excludePFMuons_ ( iConfig.getParameter<bool>("excludePFMuons") ),
  minNJetHt_      ( iConfig.getParameter<int>("minNJetHt") ),
  minNJetMht_     ( iConfig.getParameter<int>("minNJetMht") ),
  minPtJetHt_     ( iConfig.getParameter<double>("minPtJetHt") ),
  minPtJetMht_    ( iConfig.getParameter<double>("minPtJetMht") ),
  maxEtaJetHt_    ( iConfig.getParameter<double>("maxEtaJetHt") ),
  maxEtaJetMht_   ( iConfig.getParameter<double>("maxEtaJetMht") ),
  jetsLabel_      ( iConfig.getParameter<edm::InputTag>("jetsLabel") ),
  tracksLabel_    ( iConfig.getParameter<edm::InputTag>("tracksLabel") ),
  pfCandidatesLabel_ ( iConfig.getParameter<edm::InputTag>("pfCandidatesLabel") )
{
  produces<reco::METCollection>();
}


HLTHtMhtProducer::~HLTHtMhtProducer() {
}


void HLTHtMhtProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetsLabel", edm::InputTag("hltCaloJetCorrected"));
  desc.add<bool>("usePt", true);
  desc.add<int>("minNJetHt", 0);
  desc.add<int>("minNJetMht", 0);
  desc.add<double>("minPtJetHt", 40);
  desc.add<double>("minPtJetMht", 30);
  desc.add<double>("maxEtaJetHt", 3);
  desc.add<double>("maxEtaJetMht", 999);
  desc.add<bool>("useTracks", false);
  desc.add<edm::InputTag>("tracksLabel",  edm::InputTag("hltL3Muons"));
  desc.add<bool>("excludePFMuons", false);
  desc.add<edm::InputTag>("pfCandidatesLabel",  edm::InputTag("hltParticleFlow"));
  descriptions.add("hltHtMhtProducer", desc);
}


void HLTHtMhtProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  std::auto_ptr<reco::METCollection> metobject(new reco::METCollection());

  //edm::Handle<reco::CaloJetCollection> jets;
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByLabel(jetsLabel_, jets);
  edm::Handle<reco::TrackCollection> tracks;
  if (useTracks_) iEvent.getByLabel(tracksLabel_, tracks);
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  if (excludePFMuons_) iEvent.getByLabel(pfCandidatesLabel_, pfCandidates);
  
  int nj_ht = 0, nj_mht = 0;
  double ht=0.;
  double mhtx=0., mhty=0.;

  //for (reco::CaloJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); jet++) {
  for(edm::View<reco::Jet>::const_iterator jet = jets->begin(); jet != jets->end(); jet++ ) {
    double mom = (usePt_ ? jet->pt() : jet->et());
    if (mom > minPtJetHt_ and std::abs(jet->eta()) < maxEtaJetHt_) {
      ht += mom;
      ++nj_ht;
    }
    if (mom > minPtJetMht_ and std::abs(jet->eta()) < maxEtaJetMht_) {
      mhtx -= mom*cos(jet->phi());
      mhty -= mom*sin(jet->phi());
      ++nj_mht;
    }
  }
  if (useTracks_) {
    for (reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); track++) {
      if (track->pt() > minPtJetHt_ and std::abs(track->eta()) < maxEtaJetHt_) {
        ht += track->pt();
      }
      if (track->pt() > minPtJetMht_ and std::abs(track->eta()) < maxEtaJetMht_) {
        mhtx -= track->px();
        mhty -= track->py();
      }
    }
  }
  if (excludePFMuons_) {
    reco::PFCandidateCollection::const_iterator i (pfCandidates->begin());
    for (; i != pfCandidates->end(); ++i) {
      if(abs(i->pdgId())==13){
	mhtx += (i->pt())*cos(i->phi());
	mhty += (i->pt())*sin(i->phi());
      }
    }
  }

  if (nj_ht  < minNJetHt_ ) { ht = 0; }
  if (nj_mht < minNJetMht_) { mhtx = 0; mhty = 0; }

  metobject->push_back(
    reco::MET(
      ht,
      reco::MET::LorentzVector(mhtx, mhty, 0, sqrt(mhtx*mhtx + mhty*mhty)),
      reco::MET::Point()
    )
  );

  iEvent.put(metobject);
}
