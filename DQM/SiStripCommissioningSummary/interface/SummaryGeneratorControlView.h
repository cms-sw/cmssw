#ifndef DQM_SiStripCommissioningSummary_SummaryGeneratorControlView_H
#define DQM_SiStripCommissioningSummary_SummaryGeneratorControlView_H

#include "TH1.h"
#include <map>
#include <sstream>
#include <string>
#include <iostream>

// DQM common
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"

#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"

/**
   @file : DQM/SiStripCommissioningSummary/interface/SummaryGeneratorControlView.h
   @@ class : SummaryGeneratorControlView
   @@ author : M.Wingham
   @@ brief : This class inherits from SummaryGenerator.h. It implements the 
   commissioning histogram summary method according to the control view of the SST.
*/

class SummaryGeneratorControlView : public SummaryGenerator {

 public:

  /** Constructor */
  SummaryGeneratorControlView();
  
  /** Destructor */
  virtual ~SummaryGeneratorControlView();

  /** Loops through the map and fills 2 summary histograms of the stored commissioning values (one over the control view and one global). Takes the control path string of the region to be histogrammed ( in the form FecCrateA/FecSlotB/FecRingC/CcuAddrD/CcuChanE/ or any parent ) and an optional string defining what to be histogrammed (default is "values", this can also be set to "errors"), as arguments. */
  void summary(TH1F* controlSumm, TH1F* summ, const std::string& dir = "", const std::string& option = "values");

  /** Histograms the stored values/errors. */
  void histogram(TH1F*, const string& dir = "", const string& option = "values");

 private:

};

#endif // DQM_SiStripCommissioningSummary_SummaryGeneratorControlView_H
