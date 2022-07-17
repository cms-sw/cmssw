#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

class ElectronMatchedCandidateProducer : public edm::global::EDProducer<> {
public:
  explicit ElectronMatchedCandidateProducer(const edm::ParameterSet &);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::GsfElectron>> electronCollectionToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> scCollectionToken_;
  double delRMatchingCut_;
};

ElectronMatchedCandidateProducer::ElectronMatchedCandidateProducer(const edm::ParameterSet &params) {
  const edm::InputTag allelectrons("gsfElectrons");
  electronCollectionToken_ = consumes<edm::View<reco::GsfElectron>>(
      params.getUntrackedParameter<edm::InputTag>("ReferenceElectronCollection", allelectrons));
  scCollectionToken_ = consumes<edm::View<reco::Candidate>>(params.getParameter<edm::InputTag>("src"));

  delRMatchingCut_ = params.getUntrackedParameter<double>("deltaR", 0.30);

  produces<edm::PtrVector<reco::Candidate>>();
  produces<edm::RefToBaseVector<reco::Candidate>>();
}

//
// member functions
//

// ------------ method called to produce the data  ------------

void ElectronMatchedCandidateProducer::produce(edm::StreamID, edm::Event &event, const edm::EventSetup &) const {
  // Create the output collection
  auto outColRef = std::make_unique<edm::RefToBaseVector<reco::Candidate>>();
  auto outColPtr = std::make_unique<edm::PtrVector<reco::Candidate>>();

  // Read electrons
  edm::Handle<edm::View<reco::GsfElectron>> electrons;
  event.getByToken(electronCollectionToken_, electrons);

  //Read candidates
  edm::Handle<edm::View<reco::Candidate>> recoCandColl;
  event.getByToken(scCollectionToken_, recoCandColl);

  unsigned int counter = 0;

  // Loop over candidates
  for (edm::View<reco::Candidate>::const_iterator scIt = recoCandColl->begin(); scIt != recoCandColl->end();
       ++scIt, ++counter) {
    // Now loop over electrons
    for (edm::View<reco::GsfElectron>::const_iterator elec = electrons->begin(); elec != electrons->end(); ++elec) {
      reco::SuperClusterRef eSC = elec->superCluster();

      double dRval = reco::deltaR((float)eSC->eta(), (float)eSC->phi(), scIt->eta(), scIt->phi());

      if (dRval < delRMatchingCut_) {
        //outCol->push_back( *scIt );
        outColRef->push_back(recoCandColl->refAt(counter));
        outColPtr->push_back(recoCandColl->ptrAt(counter));
      }  // end if loop
    }    // end electron loop

  }  // end candidate loop

  event.put(std::move(outColRef));
  event.put(std::move(outColPtr));
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronMatchedCandidateProducer);
