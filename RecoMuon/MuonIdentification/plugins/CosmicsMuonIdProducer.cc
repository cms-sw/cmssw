#include <string>
#include <vector>

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "RecoMuon/MuonIdentification/interface/MuonCosmicsId.h"

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CosmicsMuonIdProducer : public edm::EDProducer {
public:
  CosmicsMuonIdProducer(const edm::ParameterSet& iConfig) :
    inputMuonCollection_(iConfig.getParameter<edm::InputTag>("muonCollection")),
    inputTrackCollections_(iConfig.getParameter<std::vector<edm::InputTag> >("trackCollections"))
  {
    produces<edm::ValueMap<unsigned int> >().setBranchAlias("cosmicsVeto");
  }
  virtual ~CosmicsMuonIdProducer() {}

private:
  virtual void produce(edm::Event&, const edm::EventSetup&);
  edm::InputTag inputMuonCollection_;
  std::vector<edm::InputTag> inputTrackCollections_;
};

void
CosmicsMuonIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByLabel(inputMuonCollection_, muons);
  // reserve some space
  std::vector<unsigned int> values;
  values.reserve(muons->size());
  
  for(reco::MuonCollection::const_iterator muon = muons->begin(); 
      muon != muons->end(); ++muon)
    {
      unsigned int foundPartner(0);
      if ( muon->innerTrack().isNonnull() ){
	for ( unsigned int i=0; i<inputTrackCollections_.size(); ++i )
	  {
	    edm::Handle<reco::TrackCollection> tracks;
	    iEvent.getByLabel(inputTrackCollections_.at(i), tracks);
	    if ( muonid::findOppositeTrack(tracks,*muon->innerTrack()).isNonnull() ){
	      foundPartner = i+1;
	      break;
	    }
	  }
      }
      values.push_back(foundPartner);
    }

  // create and fill value map
  std::auto_ptr<edm::ValueMap<unsigned int> > out(new edm::ValueMap<unsigned int>());
  edm::ValueMap<unsigned int>::Filler filler(*out);
  filler.insert(muons, values.begin(), values.end());
  filler.fill();

  // put value map into event
  iEvent.put(out);
}
DEFINE_FWK_MODULE(CosmicsMuonIdProducer);
