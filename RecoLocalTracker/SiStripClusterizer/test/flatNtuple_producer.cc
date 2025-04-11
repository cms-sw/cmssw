#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//#include "TH1.h"
//
// class declaration
//
//ROOT inclusion
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TTree.h"
//#include "TMath.h"
//#include "TList.h"
//#include "TString.h"

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

class flatNtuple_producer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit flatNtuple_producer(const edm::ParameterSet&);
  ~flatNtuple_producer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;  //used to select what tracks to read from configuration file
  edm::EDGetTokenT<reco::PFJetCollection> jetsToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  
  TTree* trackTree;
  edm::Service<TFileService> fs;

  edm::EventNumber_t eventN;
  constexpr static int nMax = 40000000;

  uint16_t nTracks;
  int nJets; /// uint16_t doesn't work here, give -ve number, why???
  
  edm::RunNumber_t runN;
  edm::LuminosityBlockNumber_t lumi;

  float trkPt[nMax];
  float trkEta[nMax];
  float trkPhi[nMax];
  float trkDxy1[nMax];
  float trkDxyError1[nMax];
  float trkDz1[nMax];
  float trkDzError1[nMax];
  int trkAlgo[nMax];
  int trkNHit[nMax];
  int trkNdof[nMax];
  int trkNlayer[nMax];
  float inner_xy[nMax];
  float inner_z[nMax];

  float trkChi2[nMax];
  float trkPtError[nMax];

  float jetPt[nMax];
  float jetEta[nMax];
  float jetPhi[nMax];
  float jetMass[nMax];
  
};


flatNtuple_producer::flatNtuple_producer(const edm::ParameterSet& iConfig){
  
  tracksToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
  jetsToken_ = consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("jets"));
  vertexToken_ = consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertex"));
    
  usesResource("TFileService");

  trackTree = fs->make<TTree>("tree","tree");
  trackTree->Branch("event", &eventN, "event/i");
  trackTree->Branch("run",   &runN, "run/I");
  trackTree->Branch("lumi",  &lumi, "lumi/I");
  trackTree->Branch("nTracks",  &nTracks, "nTracks/I");

  trackTree->Branch("trkPt",  trkPt, "trkPt[nTracks]/F");
  trackTree->Branch("trkEta",  trkEta, "trkEta[nTracks]/F");
  trackTree->Branch("trkPhi",  trkPhi, "trkPhi[nTracks]/F");
  trackTree->Branch("trkDxy1",  trkDxy1, "trkDxy1[nTracks]/F");
  trackTree->Branch("trkDz1",  trkDz1, "trkDz1[nTracks]/F");
  
  trackTree->Branch("trkDxyError1",  trkDxyError1, "trkDxyError1[nTracks]/F");
  trackTree->Branch("trkDzError1",  trkDzError1, "trkDzError1[nTracks]/F");
  
  trackTree->Branch("trkChi2",  trkChi2, "trkChi2[nTracks]/F");
  trackTree->Branch("trkPtError",  trkPtError, "trkPtError[nTracks]/F");

  trackTree->Branch("trkAlgo",trkAlgo,"trkAlgo[nTracks]/I");
  trackTree->Branch("trkNHit",trkNHit,"trkNHit[nTracks]/I");
  trackTree->Branch("trkNdof",trkNdof,"trkNdof[nTracks]/I");
  trackTree->Branch("trkNlayer",trkNlayer,"trkNlayer[nTracks]/I");
  trackTree->Branch("inner_xy",  inner_xy, "inner_xy[nTracks]/F");
  trackTree->Branch("inner_z",  inner_z, "inner_z[nTracks]/F");

  trackTree->Branch("nJets",  &nJets, "nJets/I");

  trackTree->Branch("jetPt",  jetPt, "jetPt[nJets]/F");
  trackTree->Branch("jetEta",  jetEta, "jetEta[nJets]/F");
  trackTree->Branch("jetPhi",  jetPhi, "jetPhi[nJets]/F");
  trackTree->Branch("jetMass",  jetMass, "jetMass[nJets]/F");
}

flatNtuple_producer::~flatNtuple_producer() = default;

// ------------ method called for each event  ------------
void flatNtuple_producer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  runN = iEvent.id().run();
  eventN = iEvent.id().event();
  lumi = iEvent.id().luminosityBlock();
  const auto& tracksHandle = iEvent.getHandle(tracksToken_);
  const auto& jetsHandle = iEvent.getHandle(jetsToken_);
  const auto& vertexHandle = iEvent.getHandle(vertexToken_);

  
  if (!tracksHandle.isValid()) {
    edm::LogError("flatNtuple_producer") << "No valid track collection found";
    return;
  }
  if (!jetsHandle.isValid()) {
    edm::LogError("flatNtuple_producer") << "No valid jet collection found";
    return;
  }
  if (!vertexHandle.isValid()) {
    edm::LogError("flatNtuple_producer") << "No valid vertex collection found";
    return;
  }
  
  // Retrieve the actual product from the handle
  const reco::TrackCollection& tracks = *tracksHandle;
  const reco::PFJetCollection& jets = *jetsHandle;
  const reco::VertexCollection vertices = *vertexHandle;

  nTracks =  tracks.size();
  for(unsigned int i=0; i<nTracks; i++)
  {
    auto track      = tracks.at(i);
    trkPt[i]        = track.pt();
    trkEta[i]       = track.eta();
    trkPhi[i]       = track.phi();
    trkDxy1[i]      = track.dxy();
    trkDz1[i]       = track.dz();
    trkDxyError1[i] = track.dxyError();
    trkDzError1[i]  = track.dzError();
    trkPtError[i]   = track.ptError();
    trkChi2[i]      = track.normalizedChi2();
    trkNdof[i]      = track.ndof();
    trkAlgo[i]      = track.algo();
    trkNHit[i]      = track.numberOfValidHits();
    trkNlayer[i]    = track.hitPattern().trackerLayersWithMeasurement();
    //const math::XYZPoint& xyz = tracks.at(i).innerPosition();
    inner_xy[i]     = track.dxy(vertices.at(0).position());
    inner_z[i]      = track.dz(vertices.at(0).position());
   // std::cout << "z " << inner_z[i] << "\t" << tracks.at(i).dz() << std::endl;
  }
  
  nJets = jets.size();
  for(int i=0; i<nJets; i++)
  {
     jetPt[i]   = jets.at(i).pt();
     jetEta[i]  = jets.at(i).eta();
     jetPhi[i]  = jets.at(i).phi();
     jetMass[i] = jets.at(i).mass();
  }
  /*
  int nvertices = vertices.size();
  float sumpt2 = 0;
  for(int i=0; i<nvertices; i++)
  {
   auto vertex = vertices.at(i);
   for (reco::Vertex::trackRef_iterator it = vertex.tracks_begin(); it != vertex.tracks_end(); it++) sumpt2 += (**it).pt()*(**it).pt();
   std::cout << sumpt2 << std::endl;
  }*/

  trackTree->Fill();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void flatNtuple_producer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  // edm::ParameterSetDescription desc;
  // desc.addUntracked<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  // descriptions.addWithDefaultLabel(desc);
  
}
//define this as a plug-in
DEFINE_FWK_MODULE(flatNtuple_producer);
