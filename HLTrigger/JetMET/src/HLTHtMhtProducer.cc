
#include "HLTrigger/JetMET/interface/HLTHtMhtProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/TrackReco/interface/Track.h"


HLTHtMhtProducer::HLTHtMhtProducer(const edm::ParameterSet & iConfig) :
  usePt_       ( iConfig.getParameter<bool>("usePt") ),
  useTracks_   ( iConfig.getParameter<bool>("useTracks") ),
  minNJet_     ( iConfig.getParameter<std::vector<int> >("minNJet") ),
  minPtJet_    ( iConfig.getParameter<std::vector<double> >("minPtJet") ),
  maxEtaJet_   ( iConfig.getParameter<std::vector<double> >("maxEtaJet") ),
  jetsLabel_   ( iConfig.getParameter<edm::InputTag>("jetsLabel") ),
  tracksLabel_ ( iConfig.getParameter< edm::InputTag >("tracksLabel") )
{

  if (minNJet_.size() != 2 or
      minPtJet_.size() != 2 or
      maxEtaJet_.size() != 2)
    edm::LogError("HLTHtMhtProducer") << "inconsistent module configuration!";

  produces<reco::METCollection>();

}


HLTHtMhtProducer::~HLTHtMhtProducer() {
}


void HLTHtMhtProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {

  std::auto_ptr<reco::METCollection> metobject(new reco::METCollection());

  edm::Handle<reco::CaloJetCollection> jets;
  iEvent.getByLabel(jetsLabel_, jets);
  edm::Handle<reco::TrackCollection> tracks;
  if (useTracks_) iEvent.getByLabel(tracksLabel_, tracks);

  int nj_ht = 0, nj_mht = 0;
  double ht=0.;
  double mhtx=0., mhty=0.;

  for (reco::CaloJetCollection::const_iterator jet = jets->begin(); jet != jets->end(); jet++) {
    double mom = (usePt_ ? jet->pt() : jet->et());
    if (mom > minPtJet_.at(0) and fabs(jet->eta()) < maxEtaJet_.at(0)) {
      ht += mom;
      ++nj_ht;
    }
    if (mom > minPtJet_.at(1) and fabs(jet->eta()) < maxEtaJet_.at(1)) {
      mhtx -= mom*cos(jet->phi());
      mhty -= mom*sin(jet->phi());
      ++nj_mht;
    }
  }
  if (useTracks_) {
    for (reco::TrackCollection::const_iterator track = tracks->begin(); track != tracks->end(); track++) {
      if (track->pt() > minPtJet_.at(0) and fabs(track->eta()) < maxEtaJet_.at(0)) {
        ht += track->pt();
      }
      if (track->pt() > minPtJet_.at(1) and fabs(track->eta()) < maxEtaJet_.at(1)) {
        mhtx -= track->px();
        mhty -= track->py();
      }
    }
  }

  metobject->push_back(
    reco::MET(
      ht,
      reco::MET::LorentzVector(mhtx, mhty, 0, sqrt(mhtx*mhtx + mhty*mhty)),
      reco::MET::Point()
    )
  );

  iEvent.put(metobject);
}
