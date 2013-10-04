// -*- c++ -*-
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

#include<memory>

class HLTTauPostProcessor : public edm::EDAnalyzer {
public:
    HLTTauPostProcessor( const edm::ParameterSet& );
    ~HLTTauPostProcessor();
    
protected:
    /// Fake Analyze
    void analyze(const edm::Event& e, const edm::EventSetup& c) ;
    
    /// EndRun
    void endRun(const edm::Run& r, const edm::EventSetup& iSetup);
    
    /// Endjob
    void endJob();
    
    void harvest();
    
private:
  std::vector<std::unique_ptr<HLTTauDQMSummaryPlotter>> summaryPlotters_;
  const bool runAtEndJob_;
  const bool runAtEndRun_;
};
