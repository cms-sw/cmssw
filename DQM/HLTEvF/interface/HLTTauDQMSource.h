/*DQM For Tau HLT
 Author : Michail Bachtis
 University of Wisconsin-Madison
 bachtis@hep.wisc.edu
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Digest.h"

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "DataFormats/Math/interface/LorentzVector.h"

//MET Includes
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

//Electron includes
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

//Muon Includes
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

//Photon Includes
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

//Track Include
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

//Plotters
#include "DQM/HLTEvF/interface/HLTTauDQML1Plotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMCaloPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMTrkPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMPathPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMLitePathPlotter.h"
#include "DQM/HLTEvF/interface/HLTTauDQMSummaryPlotter.h"

//Automatic Configuration
#include "DQM/HLTEvF/interface/HLTTauDQMAutomation.h"

//
// class declaration
//

class HLTTauDQMSource : public edm::EDAnalyzer {
public:
    HLTTauDQMSource( const edm::ParameterSet& );
    ~HLTTauDQMSource();
    
protected:
    /// BeginJob
    void beginJob();
    
    /// BeginRun
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    
    /// Fake Analyze
    void analyze(const edm::Event& e, const edm::EventSetup& c);
    
    /// Luminosity Block 
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context);
    
    /// DQM Client Diagnostic
    void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
    
    /// EndRun
    void endRun(const edm::Run& r, const edm::EventSetup& c);
    
    /// Endjob
    void endJob();
    
private:
    std::vector<edm::ParameterSet> config_;
    edm::ParameterSet matching_;
    std::string moduleName_;
    std::string hltProcessName_;
    bool hltMenuChanged_;
    edm::ParameterSet ps_;
    std::string dqmBaseFolder_;
    bool verbose_;
    
    HLTConfigProvider HLTCP_;
    HLTTauDQMAutomation automation_;
    
    //Reference
    bool doRefAnalysis_;
    edm::InputTag refTriggerEvent_;
    std::vector<edm::ParameterSet> refFilters_;
    
    int NPtBins_;
    int NEtaBins_;
    int NPhiBins_;
    double EtMax_;
    double L1MatchDr_;
    double HLTMatchDr_;
    
    //DQM prescaler
    int counterEvt_;      //counter
    int prescaleEvt_;     //every n events 
    
    //Helper function to get Trigger event primitives
    LVColl getFilterCollection(size_t index,int id,const trigger::TriggerEvent& trigEv,double);
    
    //Helper function to retrieve data from the parameter set
    void processPSet( const edm::ParameterSet& pset );
    
    //Define Dummy vectors of Plotters
    std::vector<HLTTauDQML1Plotter*> l1Plotters;
    std::vector<HLTTauDQMCaloPlotter*> caloPlotters;
    std::vector<HLTTauDQMTrkPlotter*> trackPlotters; 
    std::vector<HLTTauDQMPathPlotter*> pathPlotters;
    std::vector<HLTTauDQMLitePathPlotter*> litePathPlotters;
    std::vector<HLTTauDQMSummaryPlotter*> summaryPlotters;    
};
