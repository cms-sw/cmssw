#ifndef DQM_SiStripCommissioningSummary_CommissioningSummary_H
#define DQM_SiStripCommissioningSummary_CommissioningSummary_H

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
//@@ class : CommissioningSummary
//@@ author : M.Wingham
//@@ brief : This class is designed to contain SST commissioning values and their errors, and fill a "summary histogram" to display the information when required.
*/

class CommissioningSummary {

  public: // ----- public interface -----

  /** Constructor. Takes the summary histogram title and the monitorable granularity as arguments. */
  CommissioningSummary(string, SiStripHistoNamingScheme::Granularity);

  /** Destructor */
  virtual ~CommissioningSummary();

/**
//@@ class ReadoutId
//@@ author M.Wingham
//@@ brief : This class is designed to contain fec-key and channel number of an SST device.
*/
  class ReadoutId {

  public:

    /** Constructor: fec-key and channel number. */
    ReadoutId(unsigned int fecKey, unsigned short chan = 0) {fec_key = fecKey; channel = chan;}

    /** Destructor. */
    ~ReadoutId() {;}

    unsigned int fec_key;
    unsigned short channel;

  private:

    /** Default constructor. */
    ReadoutId() {;}
  };

  /** Updates the map, taking the ReadoutId of the device, the commissioning value and the value's error as arguments.*/
  void update(ReadoutId& readout, float, float = 0.);

  /** Returns the title of the summary histogram */
  string title();

  /** Returns the granularity of the readout devices being commissioned */
  SiStripHistoNamingScheme::Granularity granularity();

  /** Loops through the map and fills a histogram of the stored commissioning values and their errors. Each bin corresponds to one device (defined by the granularity) and is labelled with its control path i.e. fec-slot|fec-ring|ccu-address|ccu-channel(|channel). Takes the control path string of the region to be histogrammed ( in the form ControlView/FecCrateA/FecSlotB/FecRingC/CcuAddrD/CcuChanE/ or any parent ) as the argument.*/
  TH1F* controlSummary(const string& dir);

  /** Loops through the map and fills a histogram of the stored commissioning values. Takes the control path string of the control region to be histogrammed ( in the form ControlView/FecCrateA/FecSlotB/FecRingC/CcuAddrD/CcuChanE/ or any parent ) and an optional string defining what to be histogrammed (default is "values", this can also be set to "errors"), as arguments. */
  TH1F* summary(const string& dir, const string& option = "values");

  /** Returns the map, storing the commissioning value and its corresponding error for each device.*/
  map<unsigned int, map< unsigned int, pair< float,float > > >& summaryMap();
  
  private:

  /** Histogram title. */
  string title_; 

  /** Granularity of the devices being commissioned. */
  SiStripHistoNamingScheme::Granularity granularity_;

  /** A map indexed by fec-key, which holds the relevent commissioning values and their errors for each module. The map containing these values is indexed by channel number.*/  
  map<unsigned  int, map< unsigned int, pair< float,float > > > map_;

  /** Histogram of commissioning values (errors).*/ 
  TH1F* summary_;

  /** Histogram of commissioning values (errors) per device. Devices are grouped according to their control paths.*/ 
  TH1F* controlSummary_;

  /** Maximum recorded commissioning value.*/
  float max_val_;

  /** Maximum recorded commissioning error.*/
  float max_val_err_;

  /** Minimum recorded commissioning value.*/
  float min_val_;

  /** Minimum recorded commissioning error.*/
  float min_val_err_;

  /** Default constructor */
  CommissioningSummary();
  
};

inline string CommissioningSummary::title() {return title_;}

inline SiStripHistoNamingScheme::Granularity CommissioningSummary::granularity() {return granularity_;}

inline map<unsigned  int, map< unsigned int, pair< float,float > > >& CommissioningSummary::summaryMap() {return map_;}

#endif // DQM_SiStripCommissioningSummary_CommissioningSummary_H
