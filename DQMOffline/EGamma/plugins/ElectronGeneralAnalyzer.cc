
#include "DQMOffline/EGamma/plugins/ElectronGeneralAnalyzer.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

//#include "CLHEP/Units/GlobalPhysicalConstants.h"
//#include "TMath.h"

#include <iostream>

using namespace reco ;

ElectronGeneralAnalyzer::ElectronGeneralAnalyzer( const edm::ParameterSet & conf )
 : ElectronDqmAnalyzerBase(conf)
 {
  // collection input tags
  electronCollection_ = consumes<GsfElectronCollection>(conf.getParameter<edm::InputTag>("ElectronCollection"));
  matchingObjectCollection_ = consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("MatchingObjectCollection"));
  trackCollection_ = consumes<reco::TrackCollection>(conf.getParameter<edm::InputTag>("TrackCollection"));
  vertexCollection_ = consumes<reco::VertexCollection>(conf.getParameter<edm::InputTag>("VertexCollection"));
  gsftrackCollection_ = consumes<reco::GsfTrackCollection>(conf.getParameter<edm::InputTag>("GsfTrackCollection"));
  beamSpotTag_ = consumes<reco::BeamSpot>(conf.getParameter<edm::InputTag>("BeamSpot"));
  triggerResults_ = consumes<edm::TriggerResults>(conf.getParameter<edm::InputTag>("TriggerResults"));

//  // for trigger
//  HLTPathsByName_= conf.getParameter<std::vector<std::string > >("HltPaths");
//  HLTPathsByIndex_.resize(HLTPathsByName_.size());
 }

ElectronGeneralAnalyzer::~ElectronGeneralAnalyzer()
 {}

void ElectronGeneralAnalyzer::bookHistograms( DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const & )
 {
  h2_ele_beamSpotXvsY = bookH2(iBooker, "beamSpotXvsY","beam spot x vs y",100,-0.2,0.2,100,-0.2,0.2,"x (cm)","y (cm)") ;
  py_ele_nElectronsVsLs = bookP1(iBooker, "nElectronsVsLs","# gsf electrons vs LS",150,0.,1500.,0.,20.,"LS","<N_{ele}>") ;
  py_ele_nClustersVsLs = bookP1(iBooker, "nClustersVsLs","# clusters vs LS",150,0.,1500.,0.,100.,"LS","<N_{SC}>") ;
  py_ele_nGsfTracksVsLs = bookP1(iBooker, "nGsfTracksVsLs","# gsf tracks vs LS",150,0.,1500.,0.,20.,"LS","<N_{GSF tk}>") ;
  py_ele_nTracksVsLs = bookP1(iBooker, "nTracksVsLs","# tracks vs LS",150,0.,1500.,0.,100.,"LS","<N_{gen tk}>") ;
  py_ele_nVerticesVsLs = bookP1(iBooker, "nVerticesVsLs","# vertices vs LS",150,0.,1500.,0.,10.,"LS","<N_{vert}>") ;
  h1_ele_triggers = bookH1(iBooker, "triggers","hlt triggers",256,0.,256.,"HLT bit") ;
 }

void ElectronGeneralAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup & iSetup )
 {
  edm::Handle<GsfElectronCollection> gsfElectrons ;
  iEvent.getByToken(electronCollection_,gsfElectrons) ;
  edm::Handle<reco::SuperClusterCollection> recoClusters ;
  iEvent.getByToken(matchingObjectCollection_,recoClusters) ;
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(trackCollection_,tracks);
  edm::Handle<reco::GsfTrackCollection> gsfTracks;
  iEvent.getByToken(gsftrackCollection_,gsfTracks);
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vertexCollection_,vertices);
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle ;
  iEvent.getByToken(beamSpotTag_,recoBeamSpotHandle) ;
  const BeamSpot bs = *recoBeamSpotHandle ;

  edm::EventNumber_t ievt = iEvent.id().event();
  edm::RunNumber_t irun = iEvent.id().run();
  edm::LuminosityBlockNumber_t ils = iEvent.luminosityBlock();

  edm::LogInfo("ElectronGeneralAnalyzer::analyze")
    <<"Treating "<<gsfElectrons.product()->size()<<" electrons"
    <<" from event "<<ievt<<" in run "<<irun<<" and lumiblock "<<ils ;

  h2_ele_beamSpotXvsY->Fill(bs.position().x(),bs.position().y());
  py_ele_nElectronsVsLs->Fill(float(ils),(*gsfElectrons).size());
  py_ele_nClustersVsLs->Fill(float(ils),(*recoClusters).size());
  py_ele_nGsfTracksVsLs->Fill(float(ils),(*gsfTracks).size());
  py_ele_nTracksVsLs->Fill(float(ils),(*tracks).size());
  py_ele_nVerticesVsLs->Fill(float(ils),(*vertices).size());

  // trigger
  edm::Handle<edm::TriggerResults> triggerResults ;
  iEvent.getByToken(triggerResults_,triggerResults) ;
  if (triggerResults.isValid())
   {
    unsigned int i, n = triggerResults->size() ;
    for ( i=0 ; i!=n ; ++i )
     {
      if (triggerResults->accept(i))
       { h1_ele_triggers->Fill(float(i)) ; }
     }
   }
 }

