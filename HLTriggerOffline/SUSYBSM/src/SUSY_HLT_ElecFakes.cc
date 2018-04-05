#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_ElecFakes.h"
#include <iostream>

SUSY_HLT_ElecFakes::SUSY_HLT_ElecFakes(const edm::ParameterSet& ps)
{
  edm::LogInfo("SUSY_HLT_ElecFakes") << "Constructor SUSY_HLT_ElecFakes::SUSY_HLT_ElecFakes " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  HLTProcess_ = ps.getParameter<std::string>("HLTProcess");
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerFilter_ = ps.getParameter<edm::InputTag>("TriggerFilter");
  triggerJetFilter_ = ps.getParameter<edm::InputTag>("TriggerJetFilter");
}

SUSY_HLT_ElecFakes::~SUSY_HLT_ElecFakes()
{
   edm::LogInfo("SUSY_HLT_ElecFakes") << "Destructor SUSY_HLT_ElecFakes::~SUSY_HLT_ElecFakes " << std::endl;
}

void SUSY_HLT_ElecFakes::dqmBeginRun(edm::Run const &run, edm::EventSetup const &e)
{
 
  bool changed;
  
  if (!fHltConfig.init(run, e, HLTProcess_, changed)) {
    edm::LogError("SUSY_HLT_ElecFakes") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  bool pathFound = false;
  const std::vector<std::string> allTrigNames = fHltConfig.triggerNames();
  for(size_t j = 0; j <allTrigNames.size(); ++j) {
    if(allTrigNames[j].find(triggerPath_) != std::string::npos) {
      pathFound = true;
    }
  }

  if(!pathFound) {
    edm::LogInfo ("SUSY_HLT_ElecFakes") << "Path not found" << "\n";
    return;
  }

  edm::LogInfo("SUSY_HLT_ElecFakes") << "SUSY_HLT_ElecFakes::beginRun" << std::endl;
}

 void SUSY_HLT_ElecFakes::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("SUSY_HLT_ElecFakes") << "SUSY_HLT_ElecFakes::bookHistograms" << std::endl;
  //book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_ElecFakes::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
  edm::EventSetup const& context)
{
   edm::LogInfo("SUSY_HLT_ElecFakes") << "SUSY_HLT_ElecFakes::beginLuminosityBlock" << std::endl;
}



void SUSY_HLT_ElecFakes::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
  edm::LogInfo("SUSY_HLT_ElecFakes") << "SUSY_HLT_ElecFakes::analyze" << std::endl;


  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_,hltresults);
  if(!hltresults.isValid()){
    edm::LogError ("SUSY_HLT_ElecFakes") << "invalid collection: TriggerResults" << "\n";
    return;
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if(!triggerSummary.isValid()) {
    edm::LogError ("SUSY_HLT_ElecFakes") << "invalid collection: TriggerSummary" << "\n";
    return;
  }


  //get online objects
  size_t filterIndex = triggerSummary->filterIndex( triggerFilter_ );
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
  if( !(filterIndex >= triggerSummary->sizeFilters()) ){
    const trigger::Keys& keys = triggerSummary->filterKeys( filterIndex );
    for( size_t j = 0; j < keys.size(); ++j ){
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];
      //      if(foundObject.id() == 11){ //Electrons check number
	h_triggerElPt->Fill(foundObject.pt());
	h_triggerElEta->Fill(foundObject.eta());
	h_triggerElPhi->Fill(foundObject.phi());
	//      }
    }
  }
  
  filterIndex = triggerSummary->filterIndex( triggerJetFilter_ );
  //  triggerObjects = triggerSummary->getObjects();
  if( !(filterIndex >= triggerSummary->sizeFilters()) ){
    const trigger::Keys& keys = triggerSummary->filterKeys( filterIndex );
    for( size_t j = 0; j < keys.size(); ++j ){
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];
      h_triggerJetPt->Fill(foundObject.pt());
      h_triggerJetEta->Fill(foundObject.eta());
      h_triggerJetPhi->Fill(foundObject.phi());
    }
  }
  
}


void SUSY_HLT_ElecFakes::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo("SUSY_HLT_ElecFakes") << "SUSY_HLT_ElecFakes::endLuminosityBlock" << std::endl;
}


void SUSY_HLT_ElecFakes::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("SUSY_HLT_ElecFakes") << "SUSY_HLT_ElecFakes::endRun" << std::endl;
}

void SUSY_HLT_ElecFakes::bookHistos(DQMStore::IBooker & ibooker_)
{
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);
  
  //online quantities 
  h_triggerElPt  = ibooker_.book1D("triggerElPt",  "Trigger El Pt; GeV", 50, 0.0, 100.0);
  h_triggerElEta = ibooker_.book1D("triggerElEta", "Trigger El Eta", 20, -2.5, 2.5);
  h_triggerElPhi = ibooker_.book1D("triggerElPhi", "Trigger El Phi", 20, -3.5, 3.5);
  
  h_triggerJetPt  = ibooker_.book1D("triggerJetPt",  "Trigger Jet Pt; GeV", 20, 0.0, 200.0);
  h_triggerJetEta = ibooker_.book1D("triggerJetEta", "Trigger Jet Eta", 20, -3.0, 3.0);
  h_triggerJetPhi = ibooker_.book1D("triggerJetPhi", "Trigger Jet Phi", 20, -3.5, 3.5);

//  h_triggerElJetdPhi = ibooker_.book1D("triggerElJetdPhi", "Trigger El,Jet dPhi", 20, -3.5, 3.5);


  //num and den hists to be divided in harvesting step to make turn on curves
  ibooker_.cd();
}

 //define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_ElecFakes);
