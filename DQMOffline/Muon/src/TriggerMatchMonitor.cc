/*
 *  See header file for a description of this class.
 *
 *  \author Bibhuprasad Mahakud (Purdue University, West Lafayette, USA)
 */
#include "DQMOffline/Muon/interface/TriggerMatchMonitor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "TLorentzVector.h"

#include <string>
#include <TMath.h>
using namespace std;
using namespace edm;

//#define DEBUG

TriggerMatchMonitor::TriggerMatchMonitor(const edm::ParameterSet& pSet) {
  LogTrace(metname)<<"[TriggerMatchMonitor] Parameters initialization";
  
  parameters = pSet;

  // the services
  theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));
  theDbe = edm::Service<DQMStore>().operator->();
  
  beamSpotToken_ = consumes<reco::BeamSpot >(parameters.getUntrackedParameter<edm::InputTag>("offlineBeamSpot")),
  primaryVerticesToken_ = consumes<std::vector<reco::Vertex> >(parameters.getUntrackedParameter<edm::InputTag>("offlinePrimaryVertices")),
  theMuonCollectionLabel_ = consumes<edm::View<reco::Muon> >  (parameters.getParameter<edm::InputTag>("MuonCollection"));
  thePATMuonCollectionLabel_ = consumes<edm::View<pat::Muon> >  (parameters.getParameter<edm::InputTag>("patMuonCollection"));
  theVertexLabel_          = consumes<reco::VertexCollection>(parameters.getParameter<edm::InputTag>("VertexLabel"));
  theBeamSpotLabel_        = mayConsume<reco::BeamSpot>      (parameters.getParameter<edm::InputTag>("BeamSpotLabel"));
  triggerResultsToken_ = consumes<edm::TriggerResults>(parameters.getUntrackedParameter<edm::InputTag>("triggerResults"));
  triggerObjects_ = consumes<std::vector<pat::TriggerObjectStandAlone>>(parameters.getParameter<edm::InputTag>("triggerObjects"));

  // Parameters

  theFolder = parameters.getParameter<string>("folder");
}
TriggerMatchMonitor::~TriggerMatchMonitor() { 
  delete theService;
}

void TriggerMatchMonitor::bookHistograms(DQMStore::IBooker & ibooker,
					  edm::Run const & /*iRun*/,
					  edm::EventSetup const& /*iSetup*/){
    ibooker.cd();
    ibooker.setCurrentFolder(theFolder);

    // monitoring of eta parameter
    matchHists.push_back(ibooker.book1D("DelEta_HLT_IsoMu24", "DeltaEta_HLT_IsoMu24", 600, 0.0, 6.0));
    matchHists.push_back(ibooker.book1D("DelPhi_HLT_IsoMu24", "DeltaPhi_HLT_IsoMu24", 600, 0.0, 6.0));
    matchHists.push_back(ibooker.book1D("DelR_HLT_IsoMu24", "DeltaR_HLT_IsoMu24", 600, 0.0, 6.0));
    matchHists.push_back(ibooker.book1D("DelEta_L1_IsoMu24", "DeltaEta_L1_IsoMu24", 600, 0.0, 6.0));
    matchHists.push_back(ibooker.book1D("DelPhi_L1_IsoMu24", "DeltaPhi_L1_IsoMu24", 600, 0.0, 6.0));
    matchHists.push_back(ibooker.book1D("DelR_L1_IsoMu24", "DeltaR_L1_IsoMu24", 600, 0.0, 6.0));
    matchHists.push_back(ibooker.book1D("PtDiff_HLT_IsoMu24", "PtDiff_HLT_IsoMu24", 400, -500., 500.0));
    matchHists.push_back(ibooker.book1D("PtDiff_L1_IsoMu24", "PtDiff_L1_IsoMu24", 400, -500., 500.0));




}
void TriggerMatchMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  
   LogTrace(metname)<<"[TriggerMatchMonitor] Analyze the mu in different eta regions";
   theService->update(iSetup);
  
   edm::Handle<edm::View<reco::Muon> > muons; 
   iEvent.getByToken(theMuonCollectionLabel_,muons);

   edm::Handle<edm::View<pat::Muon> > PATmuons;
   iEvent.getByToken(thePATMuonCollectionLabel_,PATmuons);

   edm::Handle<std::vector<pat::TriggerObjectStandAlone> > triggerObjects;
   iEvent.getByToken(triggerObjects_, triggerObjects);

   Handle<edm::TriggerResults> triggerResults;
   iEvent.getByToken(triggerResultsToken_, triggerResults);


   reco::Vertex::Point posVtx;
   reco::Vertex::Error errVtx;
   Handle<std::vector<reco::Vertex> > recVtxs;
   iEvent.getByToken(primaryVerticesToken_,recVtxs);
   unsigned int theIndexOfThePrimaryVertex = 999.;
   for (unsigned int ind = 0; ind < recVtxs->size(); ++ind) {
     if ( (*recVtxs)[ind].isValid() && !((*recVtxs)[ind].isFake()) ) {
       theIndexOfThePrimaryVertex = ind;
       break;
     }
   }
   if (theIndexOfThePrimaryVertex<100) {
     posVtx = ((*recVtxs)[theIndexOfThePrimaryVertex]).position();
     errVtx = ((*recVtxs)[theIndexOfThePrimaryVertex]).error();
   }
   else {
     LogInfo("RecoMuonValidator") << "reco::PrimaryVertex not found, use BeamSpot position instead\n";
     Handle<reco::BeamSpot> recoBeamSpotHandle;
     iEvent.getByToken(beamSpotToken_,recoBeamSpotHandle);
     reco::BeamSpot bs = *recoBeamSpotHandle;
     posVtx = bs.position();
     errVtx(0,0) = bs.BeamWidthX();
     errVtx(1,1) = bs.BeamWidthY();
     errVtx(2,2) = bs.sigmaZ();
   }
   const reco::Vertex thePrimaryVertex(posVtx,errVtx);





   const edm::TriggerNames& trigNames = iEvent.triggerNames(*triggerResults);

   if(PATmuons.isValid()){//valid pat Muon

      for( auto & patMuon : *PATmuons){//pat muon loop
      bool Isolated=patMuon.pfIsolationR04().sumChargedHadronPt + TMath::Max(0., patMuon.pfIsolationR04().sumNeutralHadronEt + patMuon.pfIsolationR04().sumPhotonEt - 0.5*patMuon.pfIsolationR04().sumPUPt)  / patMuon.pt() < 0.25;
      

      if(patMuon.isGlobalMuon() && Isolated && patMuon.isTightMuon(thePrimaryVertex)){//isolated tight muon

      TLorentzVector offlineMuon;
      offlineMuon.SetPtEtaPhiM(patMuon.pt(), patMuon.eta(), patMuon.phi(),0.0);

      char array[] = "HLT_IsoMu24_v*";  // modifiable array of 5 bytes
      char *ptr;        // array can be modified via p
      ptr = array;      // array cannot be modified via q

      if(patMuon.triggered(ptr)){

      try{


      matchHists[0]->Fill(patMuon.eta()-patMuon.hltObject()->eta());
      matchHists[1]->Fill(patMuon.phi()-patMuon.hltObject()->phi());     
      matchHists[6]->Fill(patMuon.pt()-patMuon.hltObject()->pt());     
      TLorentzVector hltMuon;
      hltMuon.SetPtEtaPhiM(patMuon.hltObject()->pt(),patMuon.hltObject()->eta(),patMuon.hltObject()->phi(),0.0);
   
      double DelRrecoHLT=offlineMuon.DeltaR(hltMuon);
  
      matchHists[2]->Fill(DelRrecoHLT);   

      if(patMuon.l1Object() !=nullptr){
      matchHists[3]->Fill(patMuon.eta()-patMuon.l1Object()->eta());
      matchHists[4]->Fill(patMuon.phi()-patMuon.l1Object()->phi());
      
      matchHists[7]->Fill(patMuon.pt()-patMuon.l1Object()->pt());

      }

       }catch(...){}

      }

      }//isolated tight muon
      } //pat muon loop
      } //valid pat muon







 
}
