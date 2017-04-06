#include "DQMOffline/Muon/interface/MuonTiming.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include "TMath.h"
using namespace std;
using namespace edm;



MuonTiming::MuonTiming(const edm::ParameterSet& pSet) {
  parameters = pSet;

  // Input booleans
  
  // the services:
  theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));
  theMuonCollectionLabel_ = consumes<edm::View<reco::Muon> >  (parameters.getParameter<edm::InputTag>("MuonCollection"));

  tnbins = parameters.getParameter<int>("tnbins");
  tnbinsrpc = parameters.getParameter<int>("tnbinsrpc");
  terrnbins = parameters.getParameter<int>("terrnbins");
  terrnbinsrpc = parameters.getParameter<int>("terrnbinsrpc");
  ndofnbins = parameters.getParameter<int>("ndofnbins");
  ptnbins = parameters.getParameter<int>("ptnbins");
  etanbins = parameters.getParameter<int>("etanbins");
  tmax = parameters.getParameter<double>("tmax");
  tmaxrpc = parameters.getParameter<double>("tmaxrpc");
  terrmax = parameters.getParameter<double>("terrmax");
  terrmaxrpc = parameters.getParameter<double>("terrmaxrpc");
  ndofmax = parameters.getParameter<double>("ndofmax");
  ptmax = parameters.getParameter<double>("ptmax");
  etamax = parameters.getParameter<double>("etamax");
  tmin = parameters.getParameter<double>("tmin");
  tminrpc = parameters.getParameter<double>("tminrpc");
  terrmin = parameters.getParameter<double>("terrmin");
  terrminrpc = parameters.getParameter<double>("terrminrpc");
  ndofmin = parameters.getParameter<double>("ndofmin");
  ptmin = parameters.getParameter<double>("ptmin");
  etamin = parameters.getParameter<double>("etamin");
 
  etaBarrelMin = parameters.getParameter<double>("etaBarrelMin");
  etaBarrelMax = parameters.getParameter<double>("etaBarrelMax");
  etaEndcapMin = parameters.getParameter<double>("etaEndcapMin");
  etaEndcapMax = parameters.getParameter<double>("etaEndcapMax");
  etaOverlapMin = parameters.getParameter<double>("etaOverlapMin");
  etaOverlapMax = parameters.getParameter<double>("etaOverlapMax");
  
  theFolder = parameters.getParameter<string>("folder");
}


MuonTiming::~MuonTiming() {
  delete theService;
}

void MuonTiming::bookHistograms(DQMStore::IBooker & ibooker,
				      edm::Run const & /*iRun*/,
				      edm::EventSetup const & /* iSetup */){
    
  ibooker.cd();
  ibooker.setCurrentFolder(theFolder);

  EtaName.push_back("_Overlap"); EtaName.push_back("_Barrel"); EtaName.push_back("_Endcap");
  ObjectName.push_back("Sta_"); ObjectName.push_back("Glb_"); 

  for (unsigned int iEtaRegion=0; iEtaRegion<3; iEtaRegion++){
    std::vector<MonitorElement*> timeNDof_;
    std::vector<MonitorElement*> timeAtIpInOut_;
    std::vector<MonitorElement*> timeAtIpInOutRPC_;
    std::vector<MonitorElement*> timeAtIpInOutErr_;
    std::vector<MonitorElement*> timeAtIpInOutErrRPC_;
    //Only creating so far the timing information for STA muons, however the code can be extended to also Glb by just setting the limit of this loop to 2
    for(unsigned int iObjectName=0; iObjectName<1; iObjectName++) {
        timeNDof_.push_back(ibooker.book1D(ObjectName[iObjectName] + "timenDOF" + EtaName[iEtaRegion], "muon time ndof", ndofnbins, 0, ndofmax));
        timeAtIpInOut_.push_back(ibooker.book1D(ObjectName[iObjectName] + "timeAtIpInOut" + EtaName[iEtaRegion], "muon time", tnbins, tmin, tmax));
        timeAtIpInOutRPC_.push_back(ibooker.book1D(ObjectName[iObjectName] + "timeAtIpInOutRPC" + EtaName[iEtaRegion], "muon rpc time", tnbinsrpc, tminrpc, tmaxrpc));
        timeAtIpInOutErr_.push_back(ibooker.book1D(ObjectName[iObjectName] + "timeAtIpInOutErr" + EtaName[iEtaRegion], "muon time error", terrnbins, terrmin, terrmax));
        timeAtIpInOutErrRPC_.push_back(ibooker.book1D(ObjectName[iObjectName] + "timeAtIpInOutRPCErr" + EtaName[iEtaRegion], "muon rpc time error", terrnbinsrpc, terrminrpc, terrmaxrpc));
        timeNDof_[iObjectName]->setAxisTitle("Time nDof");
        timeAtIpInOut_[iObjectName]->setAxisTitle("Combined time [ns]");
        timeAtIpInOutErr_[iObjectName]->setAxisTitle("Combined time Error [ns]");
        timeAtIpInOutRPC_[iObjectName]->setAxisTitle("RPC time [ns]");
        timeAtIpInOutErrRPC_[iObjectName]->setAxisTitle("RPC time Error [ns]");
    }
    timeNDof.push_back(timeNDof_);
    timeAtIpInOut.push_back(timeAtIpInOut_);
    timeAtIpInOutRPC.push_back(timeAtIpInOutRPC_);
    timeAtIpInOutErr.push_back(timeAtIpInOutErr_);
    timeAtIpInOutErrRPC.push_back(timeAtIpInOutErrRPC_);
  }

  //Only creating so far the timing information for STA muons, however the code can be extended to also Glb by just setting the limit of this loop to 2
  for(unsigned int iObjectName=0; iObjectName<1; iObjectName++) {
    etaptVeto.push_back(ibooker.book2D(ObjectName[iObjectName] + "etapt", "Eta and Pt distribution for muons not passing the veto", ptnbins, ptmin, ptmax, etanbins, etamin, etamax));
    etaVeto.push_back(ibooker.book1D(ObjectName[iObjectName] + "eta", "Eta distribution for muons not passing the veto", etanbins, etamin, etamax));
    ptVeto.push_back(ibooker.book1D(ObjectName[iObjectName] + "pt", "Pt distribution for muons not passing the veto", ptnbins, ptmin, ptmax));
    yields.push_back(ibooker.book1D(ObjectName[iObjectName] + "yields", "Number of muons passing/not passing the different conditions", 10, 0, 10));
    yields[iObjectName]->setBinLabel(1, "Not valid time");
    yields[iObjectName]->setBinLabel(2, "Valid time");
    yields[iObjectName]->setBinLabel(3, "Not Combined time");
    yields[iObjectName]->setBinLabel(4, "Combined time");
    yields[iObjectName]->setBinLabel(5, "Not RPC time");
    yields[iObjectName]->setBinLabel(6, "RPC time");
    yields[iObjectName]->setBinLabel(7, "Combined not RPC");
    yields[iObjectName]->setBinLabel(8, "RPC not Combined");
    yields[iObjectName]->setBinLabel(9, "Not passing veto");
    yields[iObjectName]->setBinLabel(10, "Passing veto");
    etaptVeto[iObjectName]->setAxisTitle("p_{T} [GeV]");
    etaptVeto[iObjectName]->setAxisTitle("#eta#", 2);
    ptVeto[iObjectName]->setAxisTitle("p_{T} [GeV]");
    etaVeto[iObjectName]->setAxisTitle("#eta");
  }

}



void MuonTiming::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

  LogTrace(metname)<<"[MuonTiming] Analyze the mu";
  theService->update(iSetup);
 
  // Take the muon container
  edm::Handle<edm::View<reco::Muon> > muons; 
  iEvent.getByToken(theMuonCollectionLabel_,muons);
 

  if(!muons.isValid()) return;

  for (edm::View<reco::Muon>::const_iterator muon = muons->begin(); muon != muons->end(); ++muon){
    const reco::MuonTime time = muon->time(); 
    const reco::MuonTime rpcTime = muon->rpcTime();
    //Only creating so far the timing information for STA muons
    if(!muon->isStandAloneMuon() || muon->isGlobalMuon()) continue;
    reco::TrackRef track;
    //Select whether it's a global or standalone muon
    object theObject = sta;
    if(muon->isGlobalMuon()) {track = muon->combinedMuon(); theObject = glb;}
    else {track = muon->standAloneMuon(); theObject = sta;}

    //These definitions have been taken from Piotr Traczyk  
    bool cmbok =(time.nDof > 7);
    bool rpcok =(rpcTime.nDof > 1 && rpcTime.timeAtIpInOutErr == 0);
    bool veto = false;
    if (rpcok) {
       if ((fabs(rpcTime.timeAtIpInOut) > 10) && !(cmbok && fabs(time.timeAtIpInOut)<10)) veto=true;
       else if (cmbok && (time.timeAtIpInOut > 20 || time.timeAtIpInOut<-45)) veto=true;
    } 

    //std::cout << time.timeAtIpInOut << std::endl;    
    //Filling the yields histogram
    if(muon->isTimeValid()) yields[theObject]->Fill(1);
    else yields[theObject]->Fill(0);
  
    if(cmbok) yields[theObject]->Fill(3);
    else yields[theObject]->Fill(2);
    
    if(rpcok) yields[theObject]->Fill(5);
    else yields[theObject]->Fill(4);
    
    if(cmbok && !rpcok) yields[theObject]->Fill(6);
    if(!cmbok && rpcok) yields[theObject]->Fill(7);
    
    if(veto) yields[theObject]->Fill(9);
    else yields[theObject]->Fill(9);

    //Starting now with the pt and eta for vetoed and not vetoed muons
    if(veto) {
      etaptVeto[theObject]->Fill(track->pt(), track->eta());
      etaVeto[theObject]->Fill(track->eta());
      ptVeto[theObject]->Fill(track->pt());
    } 
    
    //Check the eta region of the muon
    eta theEta = barrel;
    if(fabs(track->eta()) > etaBarrelMin && fabs(track->eta()) < etaBarrelMax) theEta = barrel;
    if(fabs(track->eta()) > etaOverlapMin && fabs(track->eta()) < etaOverlapMax) theEta = overlap;
    if(fabs(track->eta()) > etaEndcapMin && fabs(track->eta()) < etaEndcapMax) theEta = endcap;
    timeNDof[theEta][theObject]->Fill(time.nDof);
    timeAtIpInOut[theEta][theObject]->Fill(time.timeAtIpInOut);
    timeAtIpInOutRPC[theEta][theObject]->Fill(rpcTime.timeAtIpInOut);
    timeAtIpInOutErr[theEta][theObject]->Fill(time.timeAtIpInOutErr);
    timeAtIpInOutErrRPC[theEta][theObject]->Fill(rpcTime.timeAtIpInOutErr);
    
  } 
}
