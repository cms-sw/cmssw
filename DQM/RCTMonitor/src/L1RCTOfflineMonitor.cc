// -*- C++ -*-
//
// Package:    L1RCTOfflineMonitor
// Class:      L1RCTOfflineMonitor
//
/**\class L1RCTOfflineMonitor L1RCTOfflineMonitor.cc DQM/RCTMonitor/src/L1RCTOfflineMonitor.cc

 Description: <one line class summary>

 Implementation:
     <notes on implementation>
*/
//
// Original Author: A. Savin
//         Created: Sat Jan 5 18:54:26 CET 2008
//     Modified by: M. Weinberg
// $Id$
//
//

// system include files
#include "DQM/RCTMonitor/interface/L1RCTOfflineMonitor.h"

//
// constructors and destructor
//
L1RCTOfflineMonitor::L1RCTOfflineMonitor(const edm::ParameterSet& iConfig):
  ecalTpgData        (iConfig.getUntrackedParameter<edm::InputTag>("ecalTpgD")),
  hcalTpgData        (iConfig.getUntrackedParameter<edm::InputTag>("hcalTpgD")),
  l1GtDaqInputSource (iConfig.getUntrackedParameter<edm::InputTag>("L1GtDaqInputTag")),
  l1GtObjectMapSource(iConfig.getUntrackedParameter<edm::InputTag>("L1GtObjectMapTag")),
  rctSourceEmulator  (iConfig.getUntrackedParameter<edm::InputTag>("rctSourceEmulator")),
  rctSourceData      (iConfig.getUntrackedParameter<edm::InputTag>("rctSourceData")),

  writeOutputFile(iConfig.getUntrackedParameter<bool>       ("WriteOutputFile")),
  outputFileName (iConfig.getUntrackedParameter<std::string>("OutputFileName")),

  nEvents(0)
{
  // now do whatever initialization is needed
  myRootfile = new TFile(outputFileName.c_str(), "RECREATE");
  myTree = new TTree("rctDqm", "Information for RCT DQM plots");
}

L1RCTOfflineMonitor::~L1RCTOfflineMonitor()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources, etc.)
}

//
// member functions
//

// -------------------- method called for each event --------------------
void L1RCTOfflineMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  FillL1RCTOfflineMonitor(iEvent, iSetup);
  nEvents++;
}

// -------------------- method called once each job just before starting event loop --------------------
void L1RCTOfflineMonitor::beginJob(const edm::EventSetup&) {BookL1RCTOfflineMonitor();}

// -------------------- method called once each job just after ending event loop --------------------
void L1RCTOfflineMonitor::endJob()
{
  myRootfile->cd();

  myTree->Write();
  delete myTree;

  myRootfile->Close();
}

void L1RCTOfflineMonitor::FillL1RCTOfflineMonitor(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get the digis
  edm::Handle<EcalTrigPrimDigiCollection> ecalTpColl;
  edm::Handle<HcalTrigPrimDigiCollection> hcalTpColl;
  edm::Handle<L1CaloEmCollection>         hwElecColl;
  edm::Handle<L1CaloEmCollection>         emulElecColl;

  iEvent.getByLabel(ecalTpgData, ecalTpColl);
  if(!ecalTpColl.isValid()) std::cout << "ECAL TP digis are not okay" << std::endl;

  iEvent.getByLabel(hcalTpgData, hcalTpColl);
  if(!hcalTpColl.isValid()) std::cout << "HCAL TP digis are not okay" << std::endl;

  iEvent.getByLabel(rctSourceData, hwElecColl);
  if(!hwElecColl.isValid()) std::cout << "Hardware digis are not okay" << std::endl;

  iEvent.getByLabel(rctSourceEmulator, emulElecColl);
  if(!emulElecColl.isValid()) std::cout << "Emulator digis are not okay" << std::endl;

  // get ECAL TP information
  nEcalTp = 0;

  for(int i = 0; i < 2400; i++)
  {
    ecalTpRank[i] = 0;
    ecalTpEta [i] = 0;
    ecalTpPhi [i] = 0;
  }

  for(EcalTrigPrimDigiCollection::const_iterator iEcalTp = ecalTpColl->begin(); iEcalTp != ecalTpColl->end(); iEcalTp++)
    if(iEcalTp->compressedEt() > 0)
    {
      ecalTpRank[nEcalTp] = iEcalTp->compressedEt();
      ecalTpEta [nEcalTp] = iEcalTp->id().ieta();
      ecalTpPhi [nEcalTp] = iEcalTp->id().iphi();

      nEcalTp++;
    }

  // get HCAL TP information
  nHcalTp = 0;

  for(int i = 0; i < 2400; i++)
  {
    hcalTpRank[i] = 0;
    hcalTpEta [i] = 0;
    hcalTpPhi [i] = 0;
  }

  for(HcalTrigPrimDigiCollection::const_iterator iHcalTp = hcalTpColl->begin(); iHcalTp != hcalTpColl->end(); iHcalTp++)
  {
    if(iHcalTp->id().iphi() >= 1  &&
       iHcalTp->id().iphi() <= 36 &&
       iHcalTp->sample(iHcalTp->presamples() - 1).compressedEt() > 0)
    {
      hcalTpRank[nHcalTp] = iHcalTp->sample(iHcalTp->presamples() - 1).compressedEt();
      hcalTpEta [nHcalTp] = iHcalTp->id().ieta();
      hcalTpPhi [nHcalTp] = iHcalTp->id().iphi();

      nHcalTp++;
    }
    else if(iHcalTp->id().iphi() >= 37 &&
	    iHcalTp->id().iphi() <= 72 &&
	    iHcalTp->sample(iHcalTp->presamples()).compressedEt() > 0)
    {
      hcalTpRank[nHcalTp] = iHcalTp->sample(iHcalTp->presamples()).compressedEt();
      hcalTpEta [nHcalTp] = iHcalTp->id().ieta();
      hcalTpPhi [nHcalTp] = iHcalTp->id().iphi();

      nHcalTp++;
    }
  }

  // get hardware electron information
  nHwElec = 0;

  for(int i = 0; i < 150; i++)
  {
    hwElecRank[i] = 0;
    hwElecEta [i] = 0;
    hwElecPhi [i] = 0;
  }

  for(L1CaloEmCollection::const_iterator iHwElec = hwElecColl->begin(); iHwElec != hwElecColl->end(); iHwElec++)
    if(iHwElec->rank() > 0)
    {
      hwElecRank[nHwElec] = int(iHwElec->rank());
      hwElecEta [nHwElec] = int(iHwElec->regionId().ieta());
      hwElecPhi [nHwElec] = int(iHwElec->regionId().iphi());

      nHwElec++;
    }

  // get emulator electron information
  nEmulElec = 0;

  for(int i = 0; i < 150; i++)
  {
    emulElecRank[i] = 0;
    emulElecEta [i] = 0;
    emulElecPhi [i] = 0;
  }

  for(L1CaloEmCollection::const_iterator iEmulElec = emulElecColl->begin(); iEmulElec != emulElecColl->end(); iEmulElec++)
    if(iEmulElec->rank() > 0)
    {
      emulElecRank[nEmulElec] = int(iEmulElec->rank());
      emulElecEta [nEmulElec] = int(iEmulElec->regionId().ieta());
      emulElecPhi [nEmulElec] = int(iEmulElec->regionId().iphi());

      nEmulElec++;
    }

  // fill the tree
  myTree->Fill();
}

void L1RCTOfflineMonitor::BookL1RCTOfflineMonitor()
{
  myTree->Branch("nEvents",   &nEvents,   "nEvents/I");
  myTree->Branch("nEcalTp",   &nEcalTp,   "nEcalTp/I");
  myTree->Branch("nHcalTp",   &nHcalTp,   "nHcalTp/I");
  myTree->Branch("nHwElec",   &nHwElec,   "nHwElec/I");
  myTree->Branch("nEmulElec", &nEmulElec, "nEmulElec/I");

  myTree->Branch("ecalTpRank", ecalTpRank, "ecalTpRank[nEcalTp]/I");
  myTree->Branch("ecalTpEta",  ecalTpEta,  "ecalTpEta [nEcalTp]/I");
  myTree->Branch("ecalTpPhi",  ecalTpPhi,  "ecalTpPhi [nEcalTp]/I");

  myTree->Branch("hcalTpRank", hcalTpRank, "hcalTpRank[nHcalTp]/I");
  myTree->Branch("hcalTpEta",  hcalTpEta,  "hcalTpEta [nHcalTp]/I");
  myTree->Branch("hcalTpPhi",  hcalTpPhi,  "hcalTpPhi [nHcalTp]/I");

  myTree->Branch("hwElecRank", hwElecRank, "hwElecRank[nHwElec]/I");
  myTree->Branch("hwElecEta",  hwElecEta,  "hwElecEta [nHwElec]/I");
  myTree->Branch("hwElecPhi",  hwElecPhi,  "hwElecPhi [nHwElec]/I");

  myTree->Branch("emulElecRank", emulElecRank, "emulElecRank[nEmulElec]/I");
  myTree->Branch("emulElecEta",  emulElecEta,  "emulElecEta [nEmulElec]/I");
  myTree->Branch("emulElecPhi",  emulElecPhi,  "emulElecPhi [nEmulElec]/I");
}

// define this as a plug-in
DEFINE_FWK_MODULE(L1RCTOfflineMonitor);
