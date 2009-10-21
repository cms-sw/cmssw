#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"

//
// class declaration
//
class EWKMuTkSelector : public edm::EDProducer {
   public:
      explicit EWKMuTkSelector(const edm::ParameterSet&);
      ~EWKMuTkSelector();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      edm::InputTag muonTag_;
      edm::InputTag trackTag_;
      double ptCutForAdditionalTracks_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

/////////////////////////////////////////////////////////////////////////////////////
EWKMuTkSelector::EWKMuTkSelector(const edm::ParameterSet& pset) {

  // What is being produced
      produces<std::vector<reco::Track> >();
      produces<std::vector<reco::Muon> >();

  // Input products
      muonTag_ = pset.getUntrackedParameter<edm::InputTag> ("MuonTag", edm::InputTag("muons"));
      trackTag_ = pset.getUntrackedParameter<edm::InputTag> ("TrackTag", edm::InputTag("gneralTracks"));
      ptCutForAdditionalTracks_ = pset.getUntrackedParameter<double> ("PtCutForAdditionalTracks");

} 

/////////////////////////////////////////////////////////////////////////////////////
EWKMuTkSelector::~EWKMuTkSelector(){
}

/////////////////////////////////////////////////////////////////////////////////////
void EWKMuTkSelector::beginJob(const edm::EventSetup&) {
}

/////////////////////////////////////////////////////////////////////////////////////
void EWKMuTkSelector::endJob(){}

/////////////////////////////////////////////////////////////////////////////////////
void EWKMuTkSelector::produce(edm::Event& ev, const edm::EventSetup&) {

      // Muon collection
      edm::Handle<edm::View<reco::Muon> > muonCollection;
      if (!ev.getByLabel(muonTag_, muonCollection)) {
            edm::LogError("") << ">>> Muon collection does not exist !!!";
            return;
      }
      unsigned int muonCollectionSize = muonCollection->size();

      // Track collection
      edm::Handle<edm::View<reco::Track> > trackCollection;
      if (!ev.getByLabel(trackTag_, trackCollection)) {
            edm::LogError("") << ">>> Track collection does not exist !!!";
            return;
      }
      unsigned int trackCollectionSize = trackCollection->size();

      // new muon and track collections
      std::auto_ptr<reco::MuonCollection> newmuons (new reco::MuonCollection);
      std::auto_ptr<reco::TrackCollection> newtracks (new reco::TrackCollection);
      reco::TrackRefProd trackRefProd = ev.getRefBeforePut<reco::TrackCollection>();

      // Select tracks for the new collection and set links in the new muon collection
      for (unsigned int j=0; j<trackCollectionSize; ++j) {
            bool alreadyWritten = false;
            reco::TrackRef tk = (trackCollection->refAt(j)).castTo<reco::TrackRef>(); 
            if (tk->pt()>ptCutForAdditionalTracks_) {
                  newtracks->push_back(*tk);
                  alreadyWritten = true;
            }
            for (unsigned int i=0; i<muonCollectionSize; ++i) {
                  const reco::Muon& mu = muonCollection->at(i);
                  if (mu.innerTrack().isNull()) continue;
                  reco::TrackRef tkInMuon = mu.innerTrack();
                  if (tk==tkInMuon) {
                        if (!alreadyWritten) {
                              newtracks->push_back(*tk);
                              alreadyWritten = true;
                        }
                        reco::Muon* newmu = mu.clone();
                        newmu->setInnerTrack(reco::TrackRef(trackRefProd,newtracks->size()-1));
                        // insert it ordered by pt
                        unsigned int newmuonCollectionSize = newmuons->size();
                        double newpt = newmu->pt();
                        bool inserted = false;
                        for (unsigned int k=0; k<newmuonCollectionSize; ++k) {
                              const reco::Muon& mu2 = newmuons->at(i);
                              if (newpt>mu2.pt()) {
                                    newmuons->insert(newmuons->begin()+k,*newmu);
                                    inserted = true;
                                    break;
                              } 
                        }
                        if (!inserted) {
                              newmuons->push_back(*newmu);
                              inserted = true;
                        }
                        break;
                  }
            }
      }

      // Write new products
      ev.put(newtracks);
      ev.put(newmuons);
}

DEFINE_FWK_MODULE(EWKMuTkSelector);
