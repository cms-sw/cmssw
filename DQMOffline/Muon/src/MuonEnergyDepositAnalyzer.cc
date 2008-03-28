
/*
 *  See header file for a description of this class.
 *
 *  $Date$
 *  $Revision$
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/MuonEnergyDepositAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;



MuonEnergyDepositAnalyzer::MuonEnergyDepositAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {

  cout<<"[MuonEnergyDepositAnalyzer] Constructor called!"<<endl;
  parameters = pSet;

}


MuonEnergyDepositAnalyzer::~MuonEnergyDepositAnalyzer() { }


void MuonEnergyDepositAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  metname = "muEnergyDepositAnalyzer";

  LogTrace(metname)<<"[MuonEnergyDepositAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("Muons/MuonEnergyDepositAnalyzer");
  std::string AlgoName = parameters.getParameter<std::string>("AlgoName");

  emNoBin = parameters.getParameter<int>("emSizeBin");
  emNoMin = parameters.getParameter<double>("emSizeMin");
  emNoMax = parameters.getParameter<double>("emSizeMax");
  std::string histname = "ecalDepositedEnergy_";
  ecalDepEnergy = dbe->book1D(histname+AlgoName, histname+AlgoName, emNoBin, emNoMin, emNoMax);

  emS9NoBin = parameters.getParameter<int>("emS9SizeBin");
  emS9NoMin = parameters.getParameter<double>("emS9SizeMin");
  emS9NoMax = parameters.getParameter<double>("emS9SizeMax");
  histname = "ecalS9DepositedEnergy_";
  ecalS9DepEnergy = dbe->book1D(histname+AlgoName, histname+AlgoName, emS9NoBin, emS9NoMin, emS9NoMax);
  
  hadNoBin = parameters.getParameter<int>("hadSizeBin");
  hadNoMin = parameters.getParameter<double>("hadSizeMin");
  hadNoMax = parameters.getParameter<double>("hadSizeMax");
  histname = "hadDepositedEnergy_";
  hcalDepEnergy = dbe->book1D(histname+AlgoName, histname+AlgoName, hadNoBin, hadNoMin, hadNoMax);

  hadS9NoBin = parameters.getParameter<int>("hadS9SizeBin");
  hadS9NoMin = parameters.getParameter<double>("hadS9SizeMin");
  hadS9NoMax = parameters.getParameter<double>("hadS9SizeMax");
  histname = "hadS9DepositedEnergy_";
  hcalS9DepEnergy = dbe->book1D(histname+AlgoName, histname+AlgoName, hadS9NoBin, hadS9NoMin, hadS9NoMax);

  hoNoBin = parameters.getParameter<int>("hoSizeBin");
  hoNoMin = parameters.getParameter<double>("hoSizeMin");
  hoNoMax = parameters.getParameter<double>("hoSizeMax");
  histname = "hoDepositedEnergy_";
  hoDepEnergy = dbe->book1D(histname+AlgoName, histname+AlgoName, hoNoBin, hoNoMin, hoNoMax);

  hoS9NoBin = parameters.getParameter<int>("hoS9SizeBin");
  hoS9NoMin = parameters.getParameter<double>("hoS9SizeMin");
  hoS9NoMax = parameters.getParameter<double>("hoS9SizeMax");
  histname = "hoS9DepositedEnergy_";
  hoS9DepEnergy = dbe->book1D(histname+AlgoName, histname+AlgoName, hoS9NoBin, hoS9NoMin, hoS9NoMax);

}



void MuonEnergyDepositAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Muon& recoMu) {

  LogTrace(metname)<<"[MuonEnergyDepositAnalyzer] Filling the histos";

  // get all the mu energy deposits
  reco::MuonEnergy muEnergy = recoMu.calEnergy();
  
  // energy deposited in ECAL
  LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.em;
  ecalDepEnergy->Fill(muEnergy.em);
  
  // energy deposited in HCAL
  LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.had;
  hcalDepEnergy->Fill(muEnergy.had);
  
  // energy deposited in HO
  LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.ho;
  hoDepEnergy->Fill(muEnergy.ho);
  
  // energy deposited in ECAL in 3*3 towers
  LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.emS9;
  ecalS9DepEnergy->Fill(muEnergy.emS9);
     
  // energy deposited in HCAL in 3*3 crystals
  LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.hadS9;
  hcalS9DepEnergy->Fill(muEnergy.hadS9);
  
  // energy deposited in HO in 3*3 crystals
  LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.hoS9;
  hoS9DepEnergy->Fill(muEnergy.hoS9);
  
}


