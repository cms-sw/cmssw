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
  const edm::ParameterSet& parameters = pSet;

  // Input booleans

  // the services:
  theMuonCollectionLabel_ = consumes<edm::View<reco::Muon> >(parameters.getParameter<edm::InputTag>("MuonCollection"));

  tnbins_ = parameters.getParameter<int>("tnbins");
  tnbinsrpc_ = parameters.getParameter<int>("tnbinsrpc");
  terrnbins_ = parameters.getParameter<int>("terrnbins");
  terrnbinsrpc_ = parameters.getParameter<int>("terrnbinsrpc");
  ndofnbins_ = parameters.getParameter<int>("ndofnbins");
  ptnbins_ = parameters.getParameter<int>("ptnbins");
  etanbins_ = parameters.getParameter<int>("etanbins");
  tmax_ = parameters.getParameter<double>("tmax");
  tmaxrpc_ = parameters.getParameter<double>("tmaxrpc");
  terrmax_ = parameters.getParameter<double>("terrmax");
  terrmaxrpc_ = parameters.getParameter<double>("terrmaxrpc");
  ndofmax_ = parameters.getParameter<double>("ndofmax");
  ptmax_ = parameters.getParameter<double>("ptmax");
  etamax_ = parameters.getParameter<double>("etamax");
  tmin_ = parameters.getParameter<double>("tmin");
  tminrpc_ = parameters.getParameter<double>("tminrpc");
  terrmin_ = parameters.getParameter<double>("terrmin");
  terrminrpc_ = parameters.getParameter<double>("terrminrpc");
  ndofmin_ = parameters.getParameter<double>("ndofmin");
  ptmin_ = parameters.getParameter<double>("ptmin");
  etamin_ = parameters.getParameter<double>("etamin");

  etaBarrelMin_ = parameters.getParameter<double>("etaBarrelMin");
  etaBarrelMax_ = parameters.getParameter<double>("etaBarrelMax");
  etaEndcapMin_ = parameters.getParameter<double>("etaEndcapMin");
  etaEndcapMax_ = parameters.getParameter<double>("etaEndcapMax");
  etaOverlapMin_ = parameters.getParameter<double>("etaOverlapMin");
  etaOverlapMax_ = parameters.getParameter<double>("etaOverlapMax");

  theFolder_ = parameters.getParameter<string>("folder");
}

MuonTiming::~MuonTiming() {}

void MuonTiming::bookHistograms(DQMStore::IBooker& ibooker,
                                edm::Run const& /*iRun*/,
                                edm::EventSetup const& /* iSetup */) {
  ibooker.cd();
  ibooker.setCurrentFolder(theFolder_);

  EtaName_.push_back("_Overlap");
  EtaName_.push_back("_Barrel");
  EtaName_.push_back("_Endcap");
  ObjectName_.push_back("Sta_");
  ObjectName_.push_back("Glb_");

  for (unsigned int iEtaRegion = 0; iEtaRegion < 3; iEtaRegion++) {
    /*std::array<MonitorElement*, 1> timeNDofv_;
    std::array<MonitorElement*, 1> timeAtIpInOutv_;
    std::array<MonitorElement*, 1> timeAtIpInOutRPCv_;
    std::array<MonitorElement*, 1> timeAtIpInOutErrv_;
    std::array<MonitorElement*, 1> timeAtIpInOutErrRPCv_;*/
    //Only creating so far the timing information for STA muons, however the code can be extended to also Glb by just setting the limit of this loop to 2
    for (unsigned int iObjectName = 0; iObjectName < 1; iObjectName++) {
      timeNDof_[iEtaRegion][iObjectName] = ibooker.book1D(
          ObjectName_[iObjectName] + "timenDOF" + EtaName_[iEtaRegion], "muon time ndof", ndofnbins_, 0, ndofmax_);
      timeAtIpInOut_[iEtaRegion][iObjectName] = ibooker.book1D(
          ObjectName_[iObjectName] + "timeAtIpInOut" + EtaName_[iEtaRegion], "muon time", tnbins_, tmin_, tmax_);
      timeAtIpInOutRPC_[iEtaRegion][iObjectName] =
          ibooker.book1D(ObjectName_[iObjectName] + "timeAtIpInOutRPC" + EtaName_[iEtaRegion],
                         "muon rpc time",
                         tnbinsrpc_,
                         tminrpc_,
                         tmaxrpc_);
      timeAtIpInOutErr_[iEtaRegion][iObjectName] =
          ibooker.book1D(ObjectName_[iObjectName] + "timeAtIpInOutErr" + EtaName_[iEtaRegion],
                         "muon time error",
                         terrnbins_,
                         terrmin_,
                         terrmax_);
      timeAtIpInOutErrRPC_[iEtaRegion][iObjectName] =
          ibooker.book1D(ObjectName_[iObjectName] + "timeAtIpInOutRPCErr" + EtaName_[iEtaRegion],
                         "muon rpc time error",
                         terrnbinsrpc_,
                         terrminrpc_,
                         terrmaxrpc_);
      timeNDof_[iEtaRegion][iObjectName]->setAxisTitle("Time nDof");
      timeAtIpInOut_[iEtaRegion][iObjectName]->setAxisTitle("Combined time [ns]");
      timeAtIpInOutErr_[iEtaRegion][iObjectName]->setAxisTitle("Combined time Error [ns]");
      timeAtIpInOutRPC_[iEtaRegion][iObjectName]->setAxisTitle("RPC time [ns]");
      timeAtIpInOutErrRPC_[iEtaRegion][iObjectName]->setAxisTitle("RPC time Error [ns]");
    }
    /*
    timeNDof_[iEtaregion] = timeNDofv_);
    timeAtIpInOut_.push_back(timeAtIpInOutv_);
    timeAtIpInOutRPC_.push_back(timeAtIpInOutRPCv_);
    timeAtIpInOutErr_.push_back(timeAtIpInOutErrv_);
    timeAtIpInOutErrRPC_.push_back(timeAtIpInOutErrRPCv_);
    */
  }

  //Only creating so far the timing information for STA muons, however the code can be extended to also Glb by just setting the limit of this loop to 2
  for (unsigned int iObjectName = 0; iObjectName < 1; iObjectName++) {
    etaptVeto_[iObjectName] = ibooker.book2D(ObjectName_[iObjectName] + "etapt",
                                             "Eta and Pt distribution for muons not passing the veto",
                                             ptnbins_,
                                             ptmin_,
                                             ptmax_,
                                             etanbins_,
                                             etamin_,
                                             etamax_);
    etaVeto_[iObjectName] = ibooker.book1D(ObjectName_[iObjectName] + "eta",
                                           "Eta distribution for muons not passing the veto",
                                           etanbins_,
                                           etamin_,
                                           etamax_);
    ptVeto_[iObjectName] = ibooker.book1D(
        ObjectName_[iObjectName] + "pt", "Pt distribution for muons not passing the veto", ptnbins_, ptmin_, ptmax_);
    yields_[iObjectName] = ibooker.book1D(
        ObjectName_[iObjectName] + "yields", "Number of muons passing/not passing the different conditions", 10, 0, 10);
    yields_[iObjectName]->setBinLabel(1, "Not valid time");
    yields_[iObjectName]->setBinLabel(2, "Valid time");
    yields_[iObjectName]->setBinLabel(3, "Not Combined time");
    yields_[iObjectName]->setBinLabel(4, "Combined time");
    yields_[iObjectName]->setBinLabel(5, "Not RPC time");
    yields_[iObjectName]->setBinLabel(6, "RPC time");
    yields_[iObjectName]->setBinLabel(7, "Combined not RPC");
    yields_[iObjectName]->setBinLabel(8, "RPC not Combined");
    yields_[iObjectName]->setBinLabel(9, "Not passing veto");
    yields_[iObjectName]->setBinLabel(10, "Passing veto");
    etaptVeto_[iObjectName]->setAxisTitle("p_{T} [GeV]");
    etaptVeto_[iObjectName]->setAxisTitle("#eta#", 2);
    ptVeto_[iObjectName]->setAxisTitle("p_{T} [GeV]");
    etaVeto_[iObjectName]->setAxisTitle("#eta");
  }
}

void MuonTiming::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LogTrace(metname_) << "[MuonTiming] Analyze the mu";

  // Take the muon container
  edm::Handle<edm::View<reco::Muon> > muons;
  iEvent.getByToken(theMuonCollectionLabel_, muons);

  if (!muons.isValid())
    return;

  for (edm::View<reco::Muon>::const_iterator muon = muons->begin(); muon != muons->end(); ++muon) {
    const reco::MuonTime time = muon->time();
    const reco::MuonTime rpcTime = muon->rpcTime();
    //Only creating so far the timing information for STA muons
    if (!muon->isStandAloneMuon() || muon->isGlobalMuon())
      continue;
    reco::TrackRef track;
    //Select whether it's a global or standalone muon
    object_ theObject = sta;
    if (muon->isGlobalMuon()) {
      track = muon->combinedMuon();
      theObject = glb;
    } else {
      track = muon->standAloneMuon();
      theObject = sta;
    }

    //These definitions have been taken from Piotr Traczyk
    bool cmbok = (time.nDof > 7);
    bool rpcok = (rpcTime.nDof > 1 && rpcTime.timeAtIpInOutErr == 0);
    bool veto = false;
    if (rpcok) {
      if ((fabs(rpcTime.timeAtIpInOut) > 10) && !(cmbok && fabs(time.timeAtIpInOut) < 10))
        veto = true;
      else if (cmbok && (time.timeAtIpInOut > 20 || time.timeAtIpInOut < -45))
        veto = true;
    }

    //std::cout << time.timeAtIpInOut << std::endl;
    //Filling the yields histogram
    if (muon->isTimeValid())
      yields_[theObject]->Fill(1);
    else
      yields_[theObject]->Fill(0);

    if (cmbok)
      yields_[theObject]->Fill(3);
    else
      yields_[theObject]->Fill(2);

    if (rpcok)
      yields_[theObject]->Fill(5);
    else
      yields_[theObject]->Fill(4);

    if (cmbok && !rpcok)
      yields_[theObject]->Fill(6);
    if (!cmbok && rpcok)
      yields_[theObject]->Fill(7);

    if (veto)
      yields_[theObject]->Fill(8);
    else
      yields_[theObject]->Fill(9);

    //Starting now with the pt and eta for vetoed and not vetoed muons
    if (veto) {
      etaptVeto_[theObject]->Fill(track->pt(), track->eta());
      etaVeto_[theObject]->Fill(track->eta());
      ptVeto_[theObject]->Fill(track->pt());
    }

    //Check the eta region of the muon
    eta_ theEta = barrel;
    if (fabs(track->eta()) >= etaBarrelMin_ && fabs(track->eta()) <= etaBarrelMax_)
      theEta = barrel;
    if (fabs(track->eta()) >= etaOverlapMin_ && fabs(track->eta()) <= etaOverlapMax_)
      theEta = overlap;
    if (fabs(track->eta()) >= etaEndcapMin_ && fabs(track->eta()) <= etaEndcapMax_)
      theEta = endcap;
    timeNDof_[theEta][theObject]->Fill(time.nDof);
    timeAtIpInOut_[theEta][theObject]->Fill(time.timeAtIpInOut);
    timeAtIpInOutRPC_[theEta][theObject]->Fill(rpcTime.timeAtIpInOut);
    timeAtIpInOutErr_[theEta][theObject]->Fill(time.timeAtIpInOutErr);
    timeAtIpInOutErrRPC_[theEta][theObject]->Fill(rpcTime.timeAtIpInOutErr);
  }
}
