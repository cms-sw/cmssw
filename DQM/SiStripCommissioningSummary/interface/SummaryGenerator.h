#ifndef DQM_SiStripCommissioningSummary_SummaryGenerator_H
#define DQM_SiStripCommissioningSummary_SummaryGenerator_H

#include <vector>
#include "TH1.h"
#include <map>
#include <sstream>
#include <string>
#include <iostream>

// DQM common
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"

using namespace std;

/**
   @file : DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h
   @@ class : SummaryGenerator
   @@ author : M.Wingham
   @@ brief : Designed to contain SST commissioning values and 
   their errors. Derived classes fill "summary histograms" to display the information when required.
*/

class SummaryGenerator {

 public: 

  /** Constructor */
  SummaryGenerator();
  
  /** Destructor */
  virtual ~SummaryGenerator();

  /** Updates the map, taking the device's key (fec/fed/det-id), the value and the value's error as arguments.*/
  void update(unsigned int, float, float = 0.);

  /** Pure virtual method. */
  virtual void summary(TH1F*, TH1F*, const string&, const string&) = 0;

  /** Histograms the stored values/errors. */
  virtual void histogram(TH1F*, const string& dir = "", const string& option = "values") = 0;
  
  protected:

  /** A map designed to holds a set of values and their errors. The map containing these values should be indexed by a device's fec/fed/readout key.*/  
  map< unsigned int, pair< float,float > > map_;

  /** Maximum recorded value.*/
  float max_val_;

  /** Maximum recorded error.*/
  float max_val_err_;

  /** Minimum recorded value.*/
  float min_val_;

  /** Minimum recorded error.*/
  float min_val_err_;
  
};

//inline map<unsigned  int, map< unsigned int, pair< float,float > > >& CommissioningSummary::summaryMap() {return map_;}

#endif // DQM_SiStripCommissioningSummary_SummaryGenerator_H
