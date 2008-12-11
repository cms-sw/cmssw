
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/10/24 16:17:18 $
 *  $Revision: 1.8 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/MuonEnergyDepositAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <string>
using namespace std;
using namespace edm;



MuonEnergyDepositAnalyzer::MuonEnergyDepositAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {

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
  std::string histname = "ecalDepositedEnergyBarrel_";
  ecalDepEnergyBarrel = dbe->book1D(histname+AlgoName, "Energy deposited in the ECAL barrel cells", emNoBin, emNoMin, emNoMax);
  ecalDepEnergyBarrel->setAxisTitle("GeV");
  histname = "ecalDepositedEnergyEndcap_";
  ecalDepEnergyEndcap = dbe->book1D(histname+AlgoName, "Energy deposited in the ECAL endcap cells", emNoBin, emNoMin, emNoMax);
  ecalDepEnergyEndcap->setAxisTitle("GeV");

  emS9NoBin = parameters.getParameter<int>("emS9SizeBin");
  emS9NoMin = parameters.getParameter<double>("emS9SizeMin");
  emS9NoMax = parameters.getParameter<double>("emS9SizeMax");
  histname = "ecalS9DepositedEnergyBarrel_";
  ecalS9DepEnergyBarrel = dbe->book1D(histname+AlgoName, "Energy deposited in the ECAL barrel 3*3 towers", emS9NoBin, emS9NoMin, emS9NoMax);
  ecalS9DepEnergyBarrel->setAxisTitle("GeV");
  histname = "ecalS9DepositedEnergyEndcap_";
  ecalS9DepEnergyEndcap = dbe->book1D(histname+AlgoName, "Energy deposited in the ECAL endcap 3*3 towers", emS9NoBin, emS9NoMin, emS9NoMax);
  ecalS9DepEnergyEndcap->setAxisTitle("GeV");
  
  hadNoBin = parameters.getParameter<int>("hadSizeBin");
  hadNoMin = parameters.getParameter<double>("hadSizeMin");
  hadNoMax = parameters.getParameter<double>("hadSizeMax");
  histname = "hadDepositedEnergyBarrel_";
  hcalDepEnergyBarrel = dbe->book1D(histname+AlgoName, "Energy deposited in the HCAL barrel cells", hadNoBin, hadNoMin, hadNoMax);
  hcalDepEnergyBarrel->setAxisTitle("GeV");
  histname = "hadDepositedEnergyEndcap_";
  hcalDepEnergyEndcap = dbe->book1D(histname+AlgoName, "Energy deposited in the HCAL endcap cells", hadNoBin, hadNoMin, hadNoMax);
  hcalDepEnergyEndcap->setAxisTitle("GeV");

  hadS9NoBin = parameters.getParameter<int>("hadS9SizeBin");
  hadS9NoMin = parameters.getParameter<double>("hadS9SizeMin");
  hadS9NoMax = parameters.getParameter<double>("hadS9SizeMax");
  histname = "hadS9DepositedEnergyBarrel_";
  hcalS9DepEnergyBarrel = dbe->book1D(histname+AlgoName, "Energy deposited in the HCAL barrel 3*3 towers", hadS9NoBin, hadS9NoMin, hadS9NoMax);
  hcalS9DepEnergyBarrel->setAxisTitle("GeV");
  histname = "hadS9DepositedEnergyEndcap_";
  hcalS9DepEnergyEndcap = dbe->book1D(histname+AlgoName, "Energy deposited in the HCAL endcap 3*3 towers", hadS9NoBin, hadS9NoMin, hadS9NoMax);
  hcalS9DepEnergyEndcap->setAxisTitle("GeV");

  hoNoBin = parameters.getParameter<int>("hoSizeBin");
  hoNoMin = parameters.getParameter<double>("hoSizeMin");
  hoNoMax = parameters.getParameter<double>("hoSizeMax");
  histname = "hoDepositedEnergy_";
  hoDepEnergy = dbe->book1D(histname+AlgoName, "Energy deposited in the HO cells", hoNoBin, hoNoMin, hoNoMax);
  hoDepEnergy->setAxisTitle("GeV");

  hoS9NoBin = parameters.getParameter<int>("hoS9SizeBin");
  hoS9NoMin = parameters.getParameter<double>("hoS9SizeMin");
  hoS9NoMax = parameters.getParameter<double>("hoS9SizeMax");
  histname = "hoS9DepositedEnergy_";
  hoS9DepEnergy = dbe->book1D(histname+AlgoName, "Energy deposited in the HO 3*3 towers", hoS9NoBin, hoS9NoMin, hoS9NoMax);
  hoS9DepEnergy->setAxisTitle("GeV");

}



void MuonEnergyDepositAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Muon& recoMu) {

  LogTrace(metname)<<"[MuonEnergyDepositAnalyzer] Filling the histos";

  // get all the mu energy deposits
  reco::MuonEnergy muEnergy = recoMu.calEnergy();
  
  // energy deposited in ECAL
  LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.em;
  if (fabs(recoMu.eta()) > 1.479) 
    ecalDepEnergyEndcap->Fill(muEnergy.em);
  else
    ecalDepEnergyBarrel->Fill(muEnergy.em);

  // energy deposited in HCAL
  LogTrace(metname) << "Energy deposited in HCAL: "<<muEnergy.had;
  if (fabs(recoMu.eta()) > 1.4)
    hcalDepEnergyEndcap->Fill(muEnergy.had);
  else
    hcalDepEnergyBarrel->Fill(muEnergy.had);
  
  // energy deposited in HO
  LogTrace(metname) << "Energy deposited in HO: "<<muEnergy.ho;
  if (fabs(recoMu.eta()) < 1.26)
    hoDepEnergy->Fill(muEnergy.ho);
  
  // energy deposited in ECAL in 3*3 towers
  LogTrace(metname) << "Energy deposited in ECAL: "<<muEnergy.emS9;
  if (fabs(recoMu.eta()) > 1.479) 
    ecalS9DepEnergyEndcap->Fill(muEnergy.em);
  else
    ecalS9DepEnergyBarrel->Fill(muEnergy.em);
     
  // energy deposited in HCAL in 3*3 crystals
  LogTrace(metname) << "Energy deposited in HCAL: "<<muEnergy.hadS9;
  if (fabs(recoMu.eta()) > 1.4)
    hcalS9DepEnergyEndcap->Fill(muEnergy.had);
  else
    hcalS9DepEnergyBarrel->Fill(muEnergy.had);
  
  // energy deposited in HO in 3*3 crystals
  LogTrace(metname) << "Energy deposited in HO: "<<muEnergy.hoS9;
  if (fabs(recoMu.eta()) < 1.26)
    hoS9DepEnergy->Fill(muEnergy.ho);
  
}


