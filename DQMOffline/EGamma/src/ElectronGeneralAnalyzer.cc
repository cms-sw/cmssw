
#include "DQMOffline/EGamma/interface/ElectronGeneralAnalyzer.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Framework/interface/TriggerNames.h"
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
  electronCollection_ = conf.getParameter<edm::InputTag>("ElectronCollection");
  matchingObjectCollection_ = conf.getParameter<edm::InputTag>("MatchingObjectCollection");
  trackCollection_ = conf.getParameter<edm::InputTag>("TrackCollection");
  vertexCollection_ = conf.getParameter<edm::InputTag>("VertexCollection");
  gsftrackCollection_ = conf.getParameter<edm::InputTag>("GsfTrackCollection");
  triggerResults_ = conf.getParameter<edm::InputTag>("TriggerResults");

//  // for trigger
//  HLTPathsByName_= conf.getParameter<std::vector<std::string > >("HltPaths");
//  HLTPathsByIndex_.resize(HLTPathsByName_.size());
 }

ElectronGeneralAnalyzer::~ElectronGeneralAnalyzer()
 {}

void ElectronGeneralAnalyzer::book()
 {
  h2_ele_beamSpotXvsY = bookH2("h2_ele_beamSpotXvsY","beam spot x vs y",100,-1.,1.,100,-1.,1.,"x (cm)","y (cm)") ;
  py_ele_nElectronsVsLs = bookP1("py_ele_nElectronsVsLs","# gsf electrons vs LS",150,0.,150.,0.,20.,"LS","<N_{ele}>") ;
  py_ele_nClustersVsLs = bookP1("py_ele_nClustersVsLs","# clusters vs LS",150,0.,150.,0.,100.,"LS","<N_{SC}>") ;
  py_ele_nGsfTracksVsLs = bookP1("py_ele_nGsfTracksVsLs","# gsf tracks vs LS",150,0.,150.,0.,20.,"LS","<N_{GSF tk}>") ;
  py_ele_nTracksVsLs = bookP1("py_ele_nTracksVsLs","# tracks vs LS",150,0.,150.,0.,100.,"LS","<N_{gen tk}>") ;
  py_ele_nVerticesVsLs = bookP1("py_ele_nVerticesVsLs","# vertices vs LS",150,0.,150.,0.,10.,"LS","<N_{vert}>") ;
  h1_ele_triggers = bookH1("h1_ele_triggers","hlt triggers",128,0.,128.,"HLT bit") ;
 }

void ElectronGeneralAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup & iSetup )
 {
  edm::Handle<GsfElectronCollection> gsfElectrons ;
  iEvent.getByLabel(electronCollection_,gsfElectrons) ;
  edm::Handle<reco::SuperClusterCollection> recoClusters ;
  iEvent.getByLabel(matchingObjectCollection_,recoClusters) ;
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(trackCollection_,tracks);
  edm::Handle<reco::GsfTrackCollection> gsfTracks;
  iEvent.getByLabel(gsftrackCollection_,gsfTracks);
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByLabel(vertexCollection_,vertices);
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle ;
  iEvent.getByType(recoBeamSpotHandle) ;
  const BeamSpot bs = *recoBeamSpotHandle ;

  int ievt = iEvent.id().event();
  int irun = iEvent.id().run();
  int ils = iEvent.luminosityBlock();

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
  iEvent.getByLabel(triggerResults_,triggerResults) ;
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

