/*DQM For Tau HLT
 Author : Michail Bachtis
 University of Wisconsin-Madison
 bachtis@hep.wisc.edu
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Digest.h"

//Plotters
#include "DQMOffline/Trigger/interface/HLTTauDQMSummaryPlotter.h"

//Automatic Configuration
#include "DQMOffline/Trigger/interface/HLTTauDQMAutomation.h"

class HLTTauPostProcessor : public edm::EDAnalyzer {
public:
    HLTTauPostProcessor( const edm::ParameterSet& );
    ~HLTTauPostProcessor();
    
protected:
    /// BeginJob
    void beginJob();
    
    /// BeginRun
    void beginRun(const edm::Run& r, const edm::EventSetup& c);
    
    /// Fake Analyze
    void analyze(const edm::Event& e, const edm::EventSetup& c) ;
    
    ///Luminosity Block 
    void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup);
    /// DQM Client Diagnostic
    void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& iSetup);
    /// EndRun
    void endRun(const edm::Run& r, const edm::EventSetup& iSetup);
    
    /// Endjob
    void endJob();
    
    void harvest();
    
private:
    //Helper function to retrieve data from the parameter set
    void processPSet( const edm::ParameterSet& pset );
    
    edm::ParameterSet ps_;
    std::vector<edm::ParameterSet> setup_;
    std::string dqmBaseFolder_;
    bool hltMenuChanged_;
    std::string hltProcessName_;
    std::string sourceModule_;
    
    double L1MatchDr_;
    double HLTMatchDr_;
    
    HLTConfigProvider HLTCP_;
    HLTTauDQMAutomation automation_;
        
    bool runAtEndJob_;
    bool runAtEndRun_;
        
    std::vector<HLTTauDQMSummaryPlotter*> summaryPlotters_;
};
