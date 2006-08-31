#ifndef DQM_SiStripCommon_SummaryGenerator_H
#define DQM_SiStripCommon_SummaryGenerator_H

#include "DQM/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "boost/cstdint.hpp"
#include "TH1.h"
#include <string>
#include <map>

/**
   @file : DQM/SiStripCommon/interface/SummaryGenerator.h
   @@ class : SummaryGenerator
   @@ author : M.Wingham
   @@ brief : Designed to contain SST commissioning values and 
   their errors. Derived classes fill "summary histograms" to display the information when required.
*/

class SummaryGenerator {

 public: 

  SummaryGenerator() { map_.clear(); }
  virtual ~SummaryGenerator() {;}

  static std::auto_ptr<SummaryGenerator> instance( sistrip::View );
  
  /** Fills the map used to generate the histogram. */
  virtual void fillMap( const std::string& top_level_dir,
			const uint32_t& key, 
			const float& value, 
			const float& error = 0. ) = 0;
  
  /** Creates a summary histogram, displaying a simple 1D distribution
      of the monitorables (stored in the map) obtained from a
      commissioning task. */
  virtual void summaryDistr( TH1& ) = 0; 
  
  /** Creates a summary histogram, displaying monitorables (stored in
      the map) obtained from a commissioning task, as a function of
      the logical struture within the tracker, view-dependent. */
  virtual void summary1D( TH1& ) = 0;

  /** Creates a summary histogram, displaying monitorables (stored in
      the map) obtained from a commissioning task, as a function of
      the logical struture within the tracker, view-dependent. */
  virtual void summary2D( TH1& ) {;}

  /** Some generic formatting of histogram. */
  static void format( const sistrip::SummaryHisto&, 
		      const sistrip::SummaryType&,
		      const sistrip::View&, 
		      const std::string& directory,
		      TH1& );

  /** Constructs the summary histogram name. */
  static std::string name( const sistrip::SummaryHisto&, 
			   const sistrip::SummaryType&,
			   const sistrip::View&, 
			   const std::string& directory );
    
  /** Returns min/max range of values stored in map. */
  std::pair<float,float> range();

  protected:

  /** A map designed to holds a set of values. The map containing
      these values should be indexed by a device's fec/fed/readout
      key.*/  
  std::map< std::string, std::pair<float,float> > map_;
  
};

#endif // DQM_SiStripCommon_SummaryGenerator_H
