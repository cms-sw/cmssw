/*
 *  See header file for a description of this class.
 *
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/interface/MuonEnergyDepositAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include <cmath>
#include <string>
using namespace std;
using namespace edm;

MuonEnergyDepositAnalyzer::MuonEnergyDepositAnalyzer(const edm::ParameterSet& pSet){
  parameters = pSet;
  
  // the services
  theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));

  theMuonCollectionLabel_ = consumes<reco::MuonCollection>(parameters.getParameter<InputTag>("MuonCollection"));

  AlgoName = parameters.getParameter<std::string>("AlgoName");
  emNoBin = parameters.getParameter<int>("emSizeBin");
  emNoMin = parameters.getParameter<double>("emSizeMin");
  emNoMax = parameters.getParameter<double>("emSizeMax");

  hadNoBin = parameters.getParameter<int>("hadSizeBin");
  hadNoMin = parameters.getParameter<double>("hadSizeMin");
  hadNoMax = parameters.getParameter<double>("hadSizeMax");
  
  emS9NoBin = parameters.getParameter<int>("emS9SizeBin");
  emS9NoMin = parameters.getParameter<double>("emS9SizeMin");
  emS9NoMax = parameters.getParameter<double>("emS9SizeMax");
  
  hadS9NoBin = parameters.getParameter<int>("hadS9SizeBin");
  hadS9NoMin = parameters.getParameter<double>("hadS9SizeMin");
  hadS9NoMax = parameters.getParameter<double>("hadS9SizeMax");

  hoNoBin = parameters.getParameter<int>("hoSizeBin");
  hoNoMin = parameters.getParameter<double>("hoSizeMin");
  hoNoMax = parameters.getParameter<double>("hoSizeMax");
  
  hoS9NoBin = parameters.getParameter<int>("hoS9SizeBin");
  hoS9NoMin = parameters.getParameter<double>("hoS9SizeMin");
  hoS9NoMax = parameters.getParameter<double>("hoS9SizeMax");
}
MuonEnergyDepositAnalyzer::~MuonEnergyDepositAnalyzer() {   
  delete theService;  
}
void MuonEnergyDepositAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					       edm::Run const & /*iRun*/,
					       edm::EventSetup const & /* iSetup */){
  ibooker.cd();
  ibooker.setCurrentFolder("Muons/MuonEnergyDepositAnalyzer");

  std::string histname = "ecalDepositedEnergyBarrel_";
  ecalDepEnergyBarrel = ibooker.book1D(histname+AlgoName, "Energy deposited in the ECAL barrel cells", emNoBin, emNoMin, emNoMax);
  ecalDepEnergyBarrel->setAxisTitle("GeV");
  
  histname = "ecalDepositedEnergyEndcap_";
  ecalDepEnergyEndcap = ibooker.book1D(histname+AlgoName, "Energy deposited in the ECAL endcap cells", emNoBin, emNoMin, emNoMax);
  ecalDepEnergyEndcap->setAxisTitle("GeV");

  histname = "ecalS9DepositedEnergyBarrel_";
  ecalS9DepEnergyBarrel = ibooker.book1D(histname+AlgoName, "Energy deposited in the ECAL barrel 3*3 towers", emS9NoBin, emS9NoMin, emS9NoMax);
  ecalS9DepEnergyBarrel->setAxisTitle("GeV");
  histname = "ecalS9DepositedEnergyEndcap_";
  ecalS9DepEnergyEndcap = ibooker.book1D(histname+AlgoName, "Energy deposited in the ECAL endcap 3*3 towers", emS9NoBin, emS9NoMin, emS9NoMax);
  ecalS9DepEnergyEndcap->setAxisTitle("GeV");
  histname = "ecalS9PointingMuDepositedEnergy_Glb_";
  ecalS9PointingMuDepEnergy_Glb = ibooker.book1D(histname+AlgoName, "Pointing glb muons energy deposited in the ECAL 3*3 towers", emS9NoBin, emS9NoMin, emS9NoMax);
  ecalS9PointingMuDepEnergy_Glb->setAxisTitle("GeV"); 
  histname = "ecalS9PointingMuDepositedEnergy_Tk_";
  ecalS9PointingMuDepEnergy_Tk = ibooker.book1D(histname+AlgoName, "Pointing tk muons energy deposited in the ECAL 3*3 towers", emS9NoBin, emS9NoMin, emS9NoMax);
  ecalS9PointingMuDepEnergy_Tk->setAxisTitle("GeV"); 
  histname = "ecalS9PointingMuDepositedEnergy_Sta_";
  ecalS9PointingMuDepEnergy_Sta = ibooker.book1D(histname+AlgoName, "Pointing sta muons energy deposited in the ECAL 3*3 towers", emS9NoBin, emS9NoMin, emS9NoMax);
  ecalS9PointingMuDepEnergy_Sta->setAxisTitle("GeV"); 
  
  histname = "hadDepositedEnergyBarrel_";
  hcalDepEnergyBarrel = ibooker.book1D(histname+AlgoName, "Energy deposited in the HCAL barrel cells", hadNoBin, hadNoMin, hadNoMax);
  hcalDepEnergyBarrel->setAxisTitle("GeV");
  histname = "hadDepositedEnergyEndcap_";
  hcalDepEnergyEndcap = ibooker.book1D(histname+AlgoName, "Energy deposited in the HCAL endcap cells", hadNoBin, hadNoMin, hadNoMax);
  hcalDepEnergyEndcap->setAxisTitle("GeV");
  
  histname = "hadS9DepositedEnergyBarrel_";
  hcalS9DepEnergyBarrel = ibooker.book1D(histname+AlgoName, "Energy deposited in the HCAL barrel 3*3 towers", hadS9NoBin, hadS9NoMin, hadS9NoMax);
  hcalS9DepEnergyBarrel->setAxisTitle("GeV");
  histname = "hadS9DepositedEnergyEndcap_";
  hcalS9DepEnergyEndcap = ibooker.book1D(histname+AlgoName, "Energy deposited in the HCAL endcap 3*3 towers", hadS9NoBin, hadS9NoMin, hadS9NoMax);
  hcalS9DepEnergyEndcap->setAxisTitle("GeV");
  histname = "hadS9PointingMuDepositedEnergy_Glb_";
  hcalS9PointingMuDepEnergy_Glb = ibooker.book1D(histname+AlgoName, "Pointing glb muons energy deposited in the HCAL endcap 3*3 towers", hadS9NoBin, hadS9NoMin, hadS9NoMax);
  hcalS9PointingMuDepEnergy_Glb->setAxisTitle("GeV");
  histname = "hadS9PointingMuDepositedEnergy_Tk_";
  hcalS9PointingMuDepEnergy_Tk = ibooker.book1D(histname+AlgoName, "Pointing tk muons energy deposited in the HCAL endcap 3*3 towers", hadS9NoBin, hadS9NoMin, hadS9NoMax);
  hcalS9PointingMuDepEnergy_Tk->setAxisTitle("GeV");
  histname = "hadS9PointingMuDepositedEnergy_Sta_";
  hcalS9PointingMuDepEnergy_Sta = ibooker.book1D(histname+AlgoName, "Pointing sta muons energy deposited in the HCAL endcap 3*3 towers", hadS9NoBin, hadS9NoMin, hadS9NoMax);
  hcalS9PointingMuDepEnergy_Sta->setAxisTitle("GeV");

  histname = "hoDepositedEnergy_";
  hoDepEnergy = ibooker.book1D(histname+AlgoName, "Energy deposited in the HO cells", hoNoBin, hoNoMin, hoNoMax);
  hoDepEnergy->setAxisTitle("GeV");

  histname = "hoS9DepositedEnergy_";
  hoS9DepEnergy = ibooker.book1D(histname+AlgoName, "Energy deposited in the HO 3*3 towers", hoS9NoBin, hoS9NoMin, hoS9NoMax);
  hoS9DepEnergy->setAxisTitle("GeV");
  histname = "hoS9PointingMuDepositedEnergy_Glb_";
  hoS9PointingMuDepEnergy_Glb = ibooker.book1D(histname+AlgoName, "Pointing glb muons energy deposited in the HO 3*3 towers", hoS9NoBin, hoS9NoMin, hoS9NoMax);
  hoS9PointingMuDepEnergy_Glb->setAxisTitle("GeV");
  histname = "hoS9PointingMuDepositedEnergy_Tk_";
  hoS9PointingMuDepEnergy_Tk = ibooker.book1D(histname+AlgoName, "Pointing tk muons energy deposited in the HO 3*3 towers", hoS9NoBin, hoS9NoMin, hoS9NoMax);
  hoS9PointingMuDepEnergy_Tk->setAxisTitle("GeV");
  histname = "hoS9PointingMuDepositedEnergy_Sta_";
  hoS9PointingMuDepEnergy_Sta = ibooker.book1D(histname+AlgoName, "Pointing sta muons energy deposited in the HO 3*3 towers", hoS9NoBin, hoS9NoMin, hoS9NoMax);
  hoS9PointingMuDepEnergy_Sta->setAxisTitle("GeV");
}
void MuonEnergyDepositAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  LogTrace(metname)<<"[MuonEnergyDepositAnalyzer] Filling the histos";
  theService->update(iSetup);
  
  // Take the muon container
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(theMuonCollectionLabel_,muons);
  
  if (!muons.isValid()) return;
  
  for (reco::MuonCollection::const_iterator recoMu = muons->begin(); recoMu!=muons->end(); ++recoMu){
    // get all the mu energy deposits
    reco::MuonEnergy muEnergy = recoMu->calEnergy();
  
    // energy deposited in ECAL
    LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.em;
    if (fabs(recoMu->eta()) > 1.479) 
      ecalDepEnergyEndcap->Fill(muEnergy.em);
    else
      ecalDepEnergyBarrel->Fill(muEnergy.em);

    // energy deposited in HCAL
    LogTrace(metname) << "Energy deposited in HCAL: "<<muEnergy.had;
    if (fabs(recoMu->eta()) > 1.4)
      hcalDepEnergyEndcap->Fill(muEnergy.had);
    else
      hcalDepEnergyBarrel->Fill(muEnergy.had);
  
    // energy deposited in HO
    LogTrace(metname) << "Energy deposited in HO: "<<muEnergy.ho;
    if (fabs(recoMu->eta()) < 1.26)
      hoDepEnergy->Fill(muEnergy.ho);
  
    // energy deposited in ECAL in 3*3 towers
    LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.emS9;
    if (fabs(recoMu->eta()) > 1.479) 
      ecalS9DepEnergyEndcap->Fill(muEnergy.emS9);
    else
      ecalS9DepEnergyBarrel->Fill(muEnergy.emS9);
     
    // energy deposited in HCAL in 3*3 crystals
    LogTrace(metname) << "Energy deposited in HCAL: "<<muEnergy.hadS9;
    if (fabs(recoMu->eta()) > 1.4)
      hcalS9DepEnergyEndcap->Fill(muEnergy.hadS9);
    else
      hcalS9DepEnergyBarrel->Fill(muEnergy.hadS9);
  
    // energy deposited in HO in 3*3 crystals
    LogTrace(metname) << "Energy deposited in HO: "<<muEnergy.hoS9;
    if (fabs(recoMu->eta()) < 1.26)
      hoS9DepEnergy->Fill(muEnergy.hoS9);
  
    // plot for energy tests
    edm::ESHandle<TransientTrackBuilder> theB;
    iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",theB);
    reco::TransientTrack TransTrack;
  
    if(recoMu->isGlobalMuon())
      TransTrack = theB->build(recoMu->globalTrack());
    if((recoMu->isTrackerMuon() || recoMu->isRPCMuon()) && !(recoMu->isGlobalMuon()))
      TransTrack = theB->build(recoMu->innerTrack());
    if(recoMu->isStandAloneMuon() && !(recoMu->isGlobalMuon()))
      TransTrack = theB->build(recoMu->outerTrack());

    TrajectoryStateOnSurface TSOS;
    TSOS = TransTrack.impactPointState();
    // section for vertex pointing muon
    if((abs(TSOS.globalPosition().z())<30) && (abs(TSOS.globalPosition().perp())<20)){
      // GLB muon
      if(recoMu->isGlobalMuon()){
	ecalS9PointingMuDepEnergy_Glb->Fill(muEnergy.emS9);
	hcalS9PointingMuDepEnergy_Glb->Fill(muEnergy.hadS9);
	hoS9PointingMuDepEnergy_Glb->Fill(muEnergy.hoS9);
      }
      // TK muon
      if(recoMu->isTrackerMuon() && !(recoMu->isGlobalMuon())){
	ecalS9PointingMuDepEnergy_Tk->Fill(muEnergy.emS9);
	hcalS9PointingMuDepEnergy_Tk->Fill(muEnergy.hadS9);
	hoS9PointingMuDepEnergy_Tk->Fill(muEnergy.hoS9);
      }
      // STA muon
      if(recoMu->isStandAloneMuon() && !(recoMu->isGlobalMuon())){
	ecalS9PointingMuDepEnergy_Sta->Fill(muEnergy.emS9);
	hcalS9PointingMuDepEnergy_Sta->Fill(muEnergy.hadS9);
	hoS9PointingMuDepEnergy_Sta->Fill(muEnergy.hoS9);
      } 
    }
  }

}


