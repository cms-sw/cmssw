#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>
#include <stdlib.h>
#include <string.h>

#include "HLTrigger/HLTanalyzers/interface/HLTHeavyIon.h"
#include "FWCore/Common/interface/TriggerNames.h"

// L1 related
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
using namespace std;

HLTHeavyIon::HLTHeavyIon(edm::ConsumesCollector && iC) {
  //set parameter defaults 
  _Monte = false;
  _Debug = false;
  _OR_BXes=false;
  UnpackBxInEvent=1;
  centralityBin_Label = edm::InputTag("centralityBin");
  centralityBin_Token = iC.consumes<int>( centralityBin_Label); 
}

void
HLTHeavyIon::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription hltParameterSet;
  desc.add<edm::ParameterSetDescription>("RunParameters",hltParameterSet);
  desc.add<bool>("Debug",false);
  desc.add<bool>("Monte",false);
  descriptions.add("hltHeavyIon", desc);
}


void HLTHeavyIon::beginRun(const edm::Run& run, const edm::EventSetup& c){ 

  bool changed(true);
  if (hltConfig_.init(run,c,processName_,changed)) {
    // if init returns TRUE, initialisation has succeeded!
    if (changed) {
      // The HLT config has actually changed wrt the previous Run, hence rebook your
      // histograms or do anything else dependent on the revised HLT config
      cout << "Initalizing HLTConfigProvider"  << endl;
    }
  } else {
    // if init returns FALSE, initialisation has NOT succeeded, which indicates a problem
    // with the file and/or code and needs to be investigated!
    cout << " HLT config extraction failure with process name " << processName_ << endl;
    // In this case, all access methods will return empty values!
  }

}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTHeavyIon::setup(const edm::ParameterSet& pSet, TTree* HltTree) {


  processName_ = pSet.getParameter<std::string>("HLTProcessName") ;

  edm::ParameterSet myHltParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  vector<std::string> parameterNames = myHltParams.getParameterNames() ;
  
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
        iParam != parameterNames.end(); iParam++ ){
    if ( (*iParam) == "Debug" ) _Debug =  myHltParams.getParameter<bool>( *iParam );
    if ( (*iParam) == "Monte" ) _Monte =  myHltParams.getParameter<bool>( *iParam );

  }

  HltEvtCnt = 0;
  const int kMaxEvtPlanes = 1000;

  fNpart = -1;
  fNcoll = -1;
  fNhard = -1;
  fPhi0 = -1;
  fb = -1;
  fNcharged = -1;
  fNchargedMR = -1;
  fMeanPt = -1;
  fMeanPtMR = -1;

  fEtMR = -1;
  fNchargedPtCut = -1;
  fNchargedPtCutMR = -1;

  nEvtPlanes = 0;
  hiBin = -1;
  hiEvtPlane = new float[kMaxEvtPlanes];

  HltTree->Branch("Npart",&fNpart,"Npart/F");
  HltTree->Branch("Ncoll",&fNcoll,"Ncoll/F");
  HltTree->Branch("Nhard",&fNhard,"Nhard/F");
  HltTree->Branch("phi0",&fPhi0,"NPhi0/F");
  HltTree->Branch("b",&fb,"b/F");
  HltTree->Branch("Ncharged",&fNcharged,"Ncharged/I");
  HltTree->Branch("NchargedMR",&fNchargedMR,"NchargedMR/I");
  HltTree->Branch("MeanPt",&fMeanPt,"MeanPt/F");
  HltTree->Branch("MeanPtMR",&fMeanPtMR,"MeanPtMR/F");
  HltTree->Branch("EtMR",&fEtMR,"EtMR/F");
  HltTree->Branch("NchargedPtCut",&fNchargedPtCut,"NchargedPtCut/I");
  HltTree->Branch("NchargedPtCutMR",&fNchargedPtCutMR,"NchargedPtCutMR/I");
  HltTree->Branch("hiBin",&hiBin,"hiBin/I");
  HltTree->Branch("hiHF",&hiHF,"hiHF/F");
  HltTree->Branch("hiHFplus",&hiHFplus,"hiHFplus/F");
  HltTree->Branch("hiHFminus",&hiHFminus,"hiHFminus/F");
  HltTree->Branch("hiZDC",&hiZDC,"hiZDC/F");
  HltTree->Branch("hiZDCplus",&hiZDCplus,"hiZDCplus/F");
  HltTree->Branch("hiZDCminus",&hiZDCminus,"hiZDCminus/F");

  HltTree->Branch("hiHFhit",&hiHFhit,"hiHFhit/F");
  HltTree->Branch("hiHFhitPlus",&hiHFhitPlus,"hiHFhitPlus/F");
  HltTree->Branch("hiHFhitMinus",&hiHFhitMinus,"hiHFhitMinus/F");

  HltTree->Branch("hiET",&hiET,"hiET/F");
  HltTree->Branch("hiEE",&hiEE,"hiEE/F");
  HltTree->Branch("hiEB",&hiEB,"hiEB/F");
  HltTree->Branch("hiEEplus",&hiEEplus,"hiEEplus/F");
  HltTree->Branch("hiEEminus",&hiEEminus,"hiEEminus/F");
  HltTree->Branch("hiNpix",&hiNpix,"hiNpix/I");
  HltTree->Branch("hiNpixelTracks",&hiNpixelTracks,"hiNpixelTracks/I");
  HltTree->Branch("hiNtracks",&hiNtracks,"hiNtracks/I");
  HltTree->Branch("hiNevtPlane",&nEvtPlanes,"hiNevtPlane/I");
  HltTree->Branch("hiEvtPlanes",hiEvtPlane,"hiEvtPlanes/F");
  HltTree->Branch("hiNtracksPtCut",&hiNtracksPtCut,"hiNtracksPtCut/I");
  HltTree->Branch("hiNtracksEtaCut",&hiNtracksEtaCut,"hiNtracksEtaCut/I");
  HltTree->Branch("hiNtracksEtaPtCut",&hiNtracksEtaPtCut,"hiNtracksEtaPtCut/I");

}

/* **Analyze the event** */
void HLTHeavyIon::analyze(const edm::Handle<edm::TriggerResults>                 & hltresults,
			  const edm::Handle<reco::Centrality>    & centrality,
			  const edm::Handle<reco::EvtPlaneCollection> & evtPlanes,
			  const edm::Handle<edm::GenHIEvent> & mc,
		      edm::EventSetup const& eventSetup,
		      edm::Event const& iEvent,
                      TTree* HltTree) {

   std::cout << " Beginning HLTHeavyIon " << std::endl;

   if(_Monte){
      fb = mc->b();
      fNpart = mc->Npart();
      fNcoll = mc->Ncoll();
      fNhard = mc->Nhard();
      fPhi0 = mc->evtPlane();
      fNcharged = mc->Ncharged();
      fNchargedMR = mc->NchargedMR();
      fMeanPt = mc->MeanPt();
      fMeanPtMR = mc->MeanPtMR();
      fEtMR = mc->EtMR();
      fNchargedPtCut = mc->NchargedPtCut();
      fNchargedPtCutMR = mc->NchargedPtCutMR();
   }

   edm::Handle<int> binHandle;
   iEvent.getByToken(centralityBin_Token,binHandle);
   hiBin = *binHandle;

  hiNpix = centrality->multiplicityPixel();
  hiNpixelTracks = centrality->NpixelTracks();
  hiNtracks = centrality->Ntracks();
  hiNtracksPtCut = centrality->NtracksPtCut();
  hiNtracksEtaCut = centrality->NtracksEtaCut();
  hiNtracksEtaPtCut = centrality->NtracksEtaPtCut();

  hiHF = centrality->EtHFtowerSum();
  hiHFplus = centrality->EtHFtowerSumPlus();
  hiHFminus = centrality->EtHFtowerSumMinus();
  hiHFhit = centrality->EtHFhitSum();
  hiHFhitPlus = centrality->EtHFhitSumPlus();
  hiHFhitMinus = centrality->EtHFhitSumMinus();

  hiZDC = centrality->zdcSum();
  hiZDCplus = centrality->zdcSumPlus();
  hiZDCminus = centrality->zdcSumMinus();

  hiEEplus = centrality->EtEESumPlus();
  hiEEminus = centrality->EtEESumMinus();
  hiEE = centrality->EtEESum();
  hiEB = centrality->EtEBSum();
  hiET = centrality->EtMidRapiditySum();

  
  if(evtPlanes.isValid()){
     nEvtPlanes = evtPlanes->size();
     for(unsigned int i = 0; i < evtPlanes->size(); ++i){
	hiEvtPlane[i] = (*evtPlanes)[i].angle();     
     }
  }
  
}
