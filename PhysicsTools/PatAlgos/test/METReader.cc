#include "PhysicsTools/PatAlgos/test/METReader.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include <TROOT.h>
#include <TVector3.h>

using namespace std;
using namespace edm;
using namespace reco;

METReader::METReader(const edm::ParameterSet& iConfig) {

  _origMetLabel = consumes<pat::METCollection>(iConfig.getParameter<edm::InputTag>("originalSlimmedMetlabel"));
  _newMetLabel = consumes<pat::METCollection>(iConfig.getParameter<edm::InputTag>("newCorrectedSlimmedMetLabel"));
  //  _newMetLabel = iConfig.getParameter<edm::InputTag>("newCorrectedSlimmedMetLabel");
  //  _t1txyMetLabel = iConfig.getParameter<edm::InputTag>("T1TxyMETLabel");


  // The root tuple ==============================
  _outputfile = iConfig.getParameter<std::string>("rootOutputFile"); 
  _file = new TFile(_outputfile.c_str(), "RECREATE");  
  _tree = new TTree("tree","tree");

  _tree->Branch("origCalo",&_origCalo);
  _tree->Branch("newRaw",&_newRaw);
  _tree->Branch("origT1",&_origT1);
  _tree->Branch("newT1",&_newT1);
  
  //  _tree->Branch("newT1TxyPhi",&_newT1TxyPhi);
  //  _tree->Branch("newT1TxyPt",&_newT1TxyPt);

  _tree->Branch("newT1Phi",&_newT1Phi);
  _tree->Branch("newT1Px",&_newT1Px);
  _tree->Branch("newT1Py",&_newT1Py);
  _tree->Branch("newT1SumEt",&_newT1SumEt);

  _tree->Branch("newT1JERUp",&_newT1JERUp);
  _tree->Branch("newT1JERDo",&_newT1JERDo);
  _tree->Branch("newT1JESUp",&_newT1JESUp);
  _tree->Branch("newT1JESDo",&_newT1JESDo);
  _tree->Branch("newT1MESUp",&_newT1MESUp);
  _tree->Branch("newT1MESDo",&_newT1MESDo);
  _tree->Branch("newT1EESUp",&_newT1EESUp);
  _tree->Branch("newT1EESDo",&_newT1EESDo);
  _tree->Branch("newT1TESUp",&_newT1TESUp);
  _tree->Branch("newT1TESDo",&_newT1TESDo);
  _tree->Branch("newT1UESUp",&_newT1UESUp);
  _tree->Branch("newT1UESDo",&_newT1UESDo);

  _n=0;
}

METReader::~METReader() { 

  _file->cd();
  _tree->Write();
  _file->Write();
  _file->Close();
}



void 
METReader::beginRun(const edm::Run& run, 
		    const edm::EventSetup & es) { }

void 
METReader::analyze(const Event& iEvent, 
		   const EventSetup& iSetup) {

  pat::METCollection origMet;
  pat::METCollection newMet;

  edm::Handle < pat::METCollection > origMetHandle;
  iEvent.getByToken(_origMetLabel,origMetHandle);
  if (origMetHandle.isValid() ) origMet = *origMetHandle;

  edm::Handle < pat::METCollection > newMetHandle;
  iEvent.getByToken(_newMetLabel,newMetHandle);
  if (newMetHandle.isValid() ) newMet = *newMetHandle;

  //============================================================

  //Raw MET ====================================================
  _origRaw = origMet[0].shiftedPt(pat::MET::NoShift, pat::MET::Raw);
  _newRaw  = newMet[0].shiftedPt(pat::MET::NoShift, pat::MET::Raw);

  //Calo MET ====================================================
  _origCalo= origMet[0].caloMETPt();

  //Type1MET ====================================================
  _origT1  = origMet[0].pt();
  _newT1   = newMet[0].pt();

  //alternate way to get the Type1 value
  _newNoShiftT1  = newMet[0].shiftedPt(pat::MET::NoShift, pat::MET::Type1); //second argument is Type1 per default

  //Type1MET uncertainties =======================================
  _newT1JERUp = newMet[0].shiftedPt(pat::MET::JetResUp);
  _newT1JERDo = newMet[0].shiftedPt(pat::MET::JetResDown);
  _newT1JESUp = newMet[0].shiftedPt(pat::MET::JetEnUp);
  _newT1JESDo = newMet[0].shiftedPt(pat::MET::JetEnDown);
  _newT1MESUp = newMet[0].shiftedPt(pat::MET::MuonEnUp);
  _newT1MESDo = newMet[0].shiftedPt(pat::MET::MuonEnDown);
  _newT1EESUp = newMet[0].shiftedPt(pat::MET::ElectronEnUp);
  _newT1EESDo = newMet[0].shiftedPt(pat::MET::ElectronEnDown);
  _newT1TESUp = newMet[0].shiftedPt(pat::MET::TauEnUp);
  _newT1TESDo = newMet[0].shiftedPt(pat::MET::TauEnDown);
  _newT1UESUp = newMet[0].shiftedPt(pat::MET::UnclusteredEnUp);
  _newT1UESDo = newMet[0].shiftedPt(pat::MET::UnclusteredEnDown);

  //other functions to access the shifted MET variables =================
  //  _newT1Phi = newMet[0].shiftedPhi(pat::MET::NoShift);  //second argument is Type1 per default
  //  _newT1Px  = newMet[0].shiftedPx(pat::MET::NoShift);  //second argument is Type1 per default
  //  _newT1Py  = newMet[0].shiftedPy(pat::MET::NoShift);  //second argument is Type1 per default
  //  _newT1SumEt = newMet[0].shiftedSumEt(pat::MET::NoShift);  //second argument is Type1 per default

  // let's store what we have in a flat tree! =============================
  _tree->Fill();

  //and some printing for people who wants numbers ========================
  if(_n<10) {
    std::cout<<"============================== New event =================================="<<std::endl;

    std::cout<<"Calo central (original collection only)-> "<<_origCalo<<std::endl;
    std::cout<<"Raw central -> original : "<<_origRaw<<"\tnew : "<<_newRaw<<std::endl;
    std::cout<<"T1 central -> original: "<<_origT1<<"\tnew : "<<_newT1<<" \t and other way to get it : "<<_newNoShiftT1<<std::endl<<std::endl;
    
    std::cout<<"-------------------- uncertainties -------------------------"<<std::endl;
    std::cout<<"T1 central -> "<<_newT1<<std::endl;
    std::cout<<"T1 JERUp   -> "<<_newT1JERUp<<std::endl;
    std::cout<<"T1 JERDo   -> "<<_newT1JERDo<<std::endl;
    std::cout<<"T1 JESUp   -> "<<_newT1JESUp<<std::endl;
    std::cout<<"T1 JESDo   -> "<<_newT1JESDo<<std::endl;
    std::cout<<"T1 MuEnUp  -> "<<_newT1MESUp<<std::endl;
    std::cout<<"T1 MuEnDo  -> "<<_newT1MESDo<<std::endl;
    std::cout<<"T1 ElEnUp  -> "<<_newT1EESUp<<std::endl;
    std::cout<<"T1 ElEnDo  -> "<<_newT1EESDo<<std::endl;
    std::cout<<"T1 UncUp   -> "<<_newT1UESUp<<std::endl;
    std::cout<<"T1 UncDo   -> "<<_newT1UESDo<<std::endl;
  }
  _n++;
}

DEFINE_FWK_MODULE(METReader);
