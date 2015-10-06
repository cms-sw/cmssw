
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "RecoMuon/MuonIdentification/plugins/ME0MuonSelector.cc"
#include "FWCore/Framework/interface/ESHandle.h"

#include <sstream>

#include <memory>
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/ME0Muon.h"
#include "DataFormats/MuonReco/interface/ME0MuonCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

class ME0MuonTrackCollProducer : public edm::stream::EDProducer<> {
public:
  explicit ME0MuonTrackCollProducer(const edm::ParameterSet&);
  //std::vector<double> findSimVtx(edm::Event& iEvent);
  ~ME0MuonTrackCollProducer();

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  edm::Handle <std::vector<reco::ME0Muon> > OurMuons;
  //edm::Handle<reco::ME0MuonCollection> muonCollectionH;
  edm::InputTag OurMuonsTag;
  std::vector<std::string> selectionTags;
  const edm::ParameterSet parset_;
  edm::EDGetTokenT<ME0MuonCollection> OurMuonsToken_;
};


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ME0MuonTrackCollProducer);


ME0MuonTrackCollProducer::ME0MuonTrackCollProducer(const edm::ParameterSet& parset) :
  OurMuonsTag(parset.getParameter<edm::InputTag>("me0MuonTag")),
  selectionTags(parset.getParameter< std::vector<std::string> >("selectionTags")),
  parset_(parset)
{
  produces<reco::TrackCollection>();
  OurMuonsToken_ = consumes<ME0MuonCollection>(OurMuonsTag);
}

ME0MuonTrackCollProducer::~ME0MuonTrackCollProducer() {
}

void ME0MuonTrackCollProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace reco;
  using namespace edm;
  Handle <ME0MuonCollection> OurMuons;
  iEvent.getByToken(OurMuonsToken_,OurMuons);

  
  std::auto_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
 
  reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
  


  for(std::vector<reco::ME0Muon>::const_iterator thismuon = OurMuons->begin();
       thismuon != OurMuons->end(); ++thismuon) {

    if (!muon::isGoodMuon(*thismuon, muon::Tight)) continue;
    reco::TrackRef trackref;    

    if (thismuon->innerTrack().isNonnull()) trackref = thismuon->innerTrack();

      const reco::Track* trk = &(*trackref);
      // pointer to old track:
      //reco::Track* newTrk = new reco::Track(*trk);

      selectedTracks->push_back( *trk );
      //selectedTrackExtras->push_back( *newExtra );
  }
  iEvent.put(selectedTracks);

}
