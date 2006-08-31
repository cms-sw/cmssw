#ifndef DQM_SiStripCommon_SummaryGeneratorControlView_H
#define DQM_SiStripCommon_SummaryGeneratorControlView_H

#include "DQM/SiStripCommon/interface/SummaryGenerator.h"

// DQM common

/**
   @file : DQM/SiStripCommon/interface/SummaryGeneratorControlView.h
   @@ class : SummaryGeneratorControlView
   @@ author : M.Wingham
   @@ brief : This class inherits from SummaryGenerator.h. It implements the 
   commissioning histogram summary method according to the control view of the SST.
*/

class SummaryGeneratorControlView : public SummaryGenerator {

 public:

  SummaryGeneratorControlView() {;}
  virtual ~SummaryGeneratorControlView() {;}

  /** Fills the map used to generate the histogram. */
  void fillMap( const std::string& directory_level,
		const uint32_t& key, 
		const float& value, 
		const float& error = 0. );
  
  /** Histograms the stored values/errors. */
  void summaryDistr( TH1& );

  /** Fills a summary histogram of the stored commissioning values
      (one over the control view and one global). Takes the control
      path string of the region to be histogrammed ( in the form
      FecCrateA/FecSlotB/FecRingC/CcuAddrD/CcuChanE/ or any parent )
      and an optional string defining what to be histogrammed (default
      is "values", this can also be set to "errors"), as arguments. */
  void summary1D( TH1& );
  
  
 private:

};

#endif // DQM_SiStripCommon_SummaryGeneratorControlView_H
