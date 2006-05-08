#ifndef DQM_SiStripCommissioningSummary_CommissioningAnalysisModule_H
#define DQM_SiStripCommissioningSummary_CommissioningAnalysisModule_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//Data Formats
#include "DataFormats/SiStripDigi/interface/Profile.h"
//#include "DataFormats/SiStripDigi/interface/Histo.h"
//analysis
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
//summary
#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummary.h"
//root
#include "TFile.h"

#include <string>

using namespace std;

/**
   @file : DQM/SiStripCommissioningSummary/interface/CommissioningAnalysisModule.h
   @class : CommissioningAnalysisModule
   @author: M.Wingham

   @brief : Plug-in module which reads TH1F commissioning histograms from the event, performs an analysis to extract "commissioning monitorables" for each device and adds them to a summary map. The contents of the map (values and errors) are then histogrammed and written to file.
*/

class CommissioningAnalysisModule : public edm::EDAnalyzer {
  
 public:
  
  /** Constructor */
  CommissioningAnalysisModule( const edm::ParameterSet& );

  /** Destructor */
  ~CommissioningAnalysisModule();
  
  /** Does nothing */
  virtual void beginJob();

  /** Fills the summary histogram and writes it to file. */
  virtual void endJob();
  
  /** Performs the analysis of each histogram in the file and saves the monitorables in the summary map. */
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  
 private:

  /** Summary objects */
  CommissioningSummary* c_summary_;
  CommissioningSummary* c_summary2_;
  string dirLevel_;
  bool controlView_;

  /** Commissioning task */
  sistrip::Task task_;

  /** Output file name */
  std::string filename_;

  /** Target gain for bias-gain task */
  double targetGain_;

  /** Run number */
  unsigned int run_;
};

#endif // DQM_SiStripCommissioningSummary_CommissioningAnalysisModule_H
