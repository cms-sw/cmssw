#include <string>
#include <vector>

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMuon/MuonIdentification/interface/MuonShowerInformationFiller.h"

class MuonShowerInformationProducer : public edm::stream::EDProducer<> {
public:
  MuonShowerInformationProducer(const edm::ParameterSet& iConfig) :
    inputMuonCollection_(iConfig.getParameter<edm::InputTag>("muonCollection")),
    inputTrackCollection_(iConfig.getParameter<edm::InputTag>("trackCollection"))
  {
    edm::ConsumesCollector iC = consumesCollector();
    showerFiller_ =  new MuonShowerInformationFiller(iConfig.getParameter<edm::ParameterSet>("ShowerInformationFillerParameters"),iC);

    muonToken_ = consumes<reco::MuonCollection>(inputMuonCollection_);
 
    produces<edm::ValueMap<reco::MuonShower> >().setBranchAlias("muonShowerInformation");
  }
   virtual ~MuonShowerInformationProducer() {
    if( showerFiller_)
      delete showerFiller_;
  }

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  edm::InputTag inputMuonCollection_;
  edm::InputTag inputTrackCollection_;
  edm::EDGetTokenT<reco::MuonCollection> muonToken_;



  MuonShowerInformationFiller *showerFiller_;
};

void
MuonShowerInformationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muonToken_, muons);

  // reserve some space for output
  std::vector<reco::MuonShower> showerInfoValues;
  showerInfoValues.reserve(muons->size());
  
  for(reco::MuonCollection::const_iterator muon = muons->begin(); 
      muon != muons->end(); ++muon)
    {
     // if (!muon->isGlobalMuon() && !muon->isStandAloneMuon()) continue;
      showerInfoValues.push_back(showerFiller_->fillShowerInformation(*muon,iEvent,iSetup));
    }

  // create and fill value map
  std::auto_ptr<edm::ValueMap<reco::MuonShower> > outC(new edm::ValueMap<reco::MuonShower>());
  edm::ValueMap<reco::MuonShower>::Filler fillerC(*outC);
  fillerC.insert(muons, showerInfoValues.begin(), showerInfoValues.end());
  fillerC.fill();

  // put value map into event
  iEvent.put(outC);
}
DEFINE_FWK_MODULE(MuonShowerInformationProducer);
