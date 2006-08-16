#ifndef DQM_SiStripCommissioningSummary_SiStripSummary_H
#define DQM_SiStripCommissioningSummary_SiStripSummary_H

#include "TH1F.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommissioningSummary/interface/SiStripControlSummaryGenerator.h"
#include "DQM/SiStripCommissioningSummary/interface/SiStripReadoutSummaryGenerator.h"

/**
   @file : DQM/SiStripCommissioningSummary/interface/SiStripSummary.h
   @class : SiStripSummary
   @author: M.Wingham

   @brief : Base class for SST commissioning summaries. Produces 2 histograms - 
   1) global summary of commissioning values/errors
   2) summary of commissioning values/errors with one bin per commissioning 
   "device". Each device is defined by a key (fec-key, readout-key or det-id)
   which are described in DataFormats/SiStripDetId.
*/


class SiStripSummary {

 public:

  /** Constructor */
  SiStripSummary(sistrip::View);
  
  /** Destructor */
  virtual ~SiStripSummary();

  /** Adds a value and and error to the summary map. The key is fec-key, readout-key or det-id depending on the view. */
  void update(unsigned int key, float value, float error = 0.);
  
  /** Fills the histogram_ member with the "values" or "errors" stored in the map. Fills summary_ with the "values" or "errors" stored - each bin represents one device and is labelled with its control/readout/detector path. Both histograms only include information on devices at or below the directory level specified.*/
  void histogram(const string& dir, const string& option = "values");
  
  /** Retrieves the summary_ data member - containing the device-level summary.*/
  TH1F* const getSummary() const;

  /** Retrieves the histogram_ data member - containing the global summary.*/
  TH1F* const getHistogram() const;

  /** Sets both histogram names. Appends the readout view onto the end of the device-level summary.*/
  void setName(const char*);

  /** Sets both histogram titles. Appends the readout view onto the end of the device-level summary. */
  void setTitle(const char*);

 protected:

  /** User-specific histogram formatting. No implementation - should be added in derived classes.*/
  virtual void format();
  
  /** The device-level summary */
  TH1F* summary_;
  
  /** The global summary */
  TH1F* histogram_;
  
  /** Holds the map and fills the histograms */
  SiStripSummaryGenerator* generator_;
  
  /** Readout view */
  sistrip::View view_;
  
 private:
  
  /** Private default constructor */
  SiStripSummary();
  
};

//------------------------------------------------------------------------------

inline TH1F* const SiStripSummary::getSummary() const { return summary_; } 

//------------------------------------------------------------------------------

inline TH1F* const SiStripSummary::getHistogram() const { return histogram_; }

//------------------------------------------------------------------------------

inline void SiStripSummary::format() {;}

//------------------------------------------------------------------------------

#endif // DQM_SiStripCommissioningSummary_SiStripSummary_H
