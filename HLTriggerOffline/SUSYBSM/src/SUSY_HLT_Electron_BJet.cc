#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_Electron_BJet.h"

SUSY_HLT_Electron_BJet::SUSY_HLT_Electron_BJet(const edm::ParameterSet& ps)
{
    edm::LogInfo("SUSY_HLT_Electron_BJet") << "Constructor SUSY_HLT_Electron_BJet::SUSY_HLT_Electron_BJet " << std::endl;
    // Get parameters from configuration file
    theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
    theElectronCollection_ = consumes<reco::GsfElectronCollection>(ps.getParameter<edm::InputTag>("ElectronCollection"));
    thePfJetCollection_ = consumes<reco::PFJetCollection>(ps.getParameter<edm::InputTag>("pfJetCollection"));
    theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
    triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
    HLTProcess_ = ps.getParameter<std::string>("HLTProcess");
    triggerPath_ = ps.getParameter<std::string>("TriggerPath");
    triggerFilterEle_ = ps.getParameter<edm::InputTag>("TriggerFilterEle");
	triggerFilterJet_ = ps.getParameter<edm::InputTag>("TriggerFilterJet");
    ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet");
    etaThrJet_ = ps.getUntrackedParameter<double>("EtaThrJet");
}

SUSY_HLT_Electron_BJet::~SUSY_HLT_Electron_BJet()
{
    edm::LogInfo("SUSY_HLT_Electron_BJet") << "Destructor SUSY_HLT_Electron_BJet::~SUSY_HLT_Electron_BJet " << std::endl;
}

void SUSY_HLT_Electron_BJet::dqmBeginRun(edm::Run const &run, edm::EventSetup const &e)
{
    
    bool changed;
    
    if (!fHltConfig.init(run, e, HLTProcess_, changed)) {
        edm::LogError("SUSY_HLT_Electron_BJet") << "Initialization of HLTConfigProvider failed!!";
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
        LogDebug("SUSY_HLT_Electron_BJet") << "Path not found" << "\n";
        return;
    }
    
    edm::LogInfo("SUSY_HLT_Electron_BJet") << "SUSY_HLT_Electron_BJet::beginRun" << std::endl;
}

void SUSY_HLT_Electron_BJet::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
    edm::LogInfo("SUSY_HLT_Electron_BJet") << "SUSY_HLT_Electron_BJet::bookHistograms" << std::endl;
    //book at beginRun
    bookHistos(ibooker_);
}

void SUSY_HLT_Electron_BJet::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
                                                        edm::EventSetup const& context)
{
    edm::LogInfo("SUSY_HLT_Electron_BJet") << "SUSY_HLT_Electron_BJet::beginLuminosityBlock" << std::endl;
}



void SUSY_HLT_Electron_BJet::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
    edm::LogInfo("SUSY_HLT_Electron_BJet") << "SUSY_HLT_Electron_BJet::analyze" << std::endl;
    
    //-------------------------------
    //--- Trigger
    //-------------------------------
    edm::Handle<edm::TriggerResults> hltresults;
    e.getByToken(triggerResults_,hltresults);
    if(!hltresults.isValid()){
        edm::LogError ("SUSY_HLT_Electron_BJet") << "invalid collection: TriggerResults" << "\n";
        return;
    }
    edm::Handle<trigger::TriggerEvent> triggerSummary;
    e.getByToken(theTrigSummary_, triggerSummary);
    if(!triggerSummary.isValid()) {
        edm::LogError ("SUSY_HLT_Electron_BJet") << "invalid collection: TriggerSummary" << "\n";
        return;
    }
    
    
    //get online objects
    std::vector<float> ptElectron;//, etaElectron, phiElectron;
    size_t filterIndex = triggerSummary->filterIndex( triggerFilterEle_ );
    trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
    if( !(filterIndex >= triggerSummary->sizeFilters()) ){
        const trigger::Keys& keys = triggerSummary->filterKeys( filterIndex );
        for( size_t j = 0; j < keys.size(); ++j ){
            trigger::TriggerObject foundObject = triggerObjects[keys[j]];
            if(fabs(foundObject.id()) == 11){ //It's an electron
			
				bool same= false;
				for(unsigned int x=0;x<ptElectron.size();x++){
					if(fabs(ptElectron[x] - foundObject.pt()) < 0.01)
						same = true;
				}
				
				if(!same){
                	h_triggerElePt->Fill(foundObject.pt());
                	h_triggerEleEta->Fill(foundObject.eta());
                	h_triggerElePhi->Fill(foundObject.phi());
                	ptElectron.push_back(foundObject.pt());
				}
            }
        }
    }
    
    size_t filterIndex2 = triggerSummary->filterIndex( triggerFilterJet_ );
    if( !(filterIndex2 >= triggerSummary->sizeFilters()) ){
        const trigger::Keys& keys = triggerSummary->filterKeys( filterIndex2 );
        for( size_t j = 0; j < keys.size(); ++j ){
            trigger::TriggerObject foundObject = triggerObjects[keys[j]];
                h_triggerJetPt->Fill(foundObject.pt());
                h_triggerJetEta->Fill(foundObject.eta());
                h_triggerJetPhi->Fill(foundObject.phi());
        }
    }
	
    
}


void SUSY_HLT_Electron_BJet::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
    edm::LogInfo("SUSY_HLT_Electron_BJet") << "SUSY_HLT_Electron_BJet::endLuminosityBlock" << std::endl;
}


void SUSY_HLT_Electron_BJet::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
    edm::LogInfo("SUSY_HLT_Electron_BJet") << "SUSY_HLT_Electron_BJet::endRun" << std::endl;
}

void SUSY_HLT_Electron_BJet::bookHistos(DQMStore::IBooker & ibooker_)
{
    ibooker_.cd();
    ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);
    
    //offline quantities
    
    //online quantities
    h_triggerElePt = ibooker_.book1D("triggerElePt", "Trigger Electron Pt; GeV", 50, 0.0, 500.0);
    h_triggerEleEta = ibooker_.book1D("triggerEleEta", "Trigger Electron Eta", 20, -3.0, 3.0);
    h_triggerElePhi = ibooker_.book1D("triggerElePhi", "Trigger Electron Phi", 20, -3.5, 3.5);
	
	h_triggerJetPt = ibooker_.book1D("triggerJetPt", "Trigger Jet Pt; GeV", 50, 0.0, 500.0);
    h_triggerJetEta = ibooker_.book1D("triggerJetEta", "Trigger Jet Eta", 20, -3.0, 3.0);
    h_triggerJetPhi = ibooker_.book1D("triggerJetPhi", "Trigger Jet Phi", 20, -3.5, 3.5);

    
    ibooker_.cd();
}

//define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_Electron_BJet);
