#ifndef DQM_SiStripCommon_SummaryGenerator_H
#define DQM_SiStripCommon_SummaryGenerator_H

#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DQM/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "boost/cstdint.hpp"
#include <string>
#include <vector>
#include <map>

class TH1;

/**
   @file DQM/SiStripCommon/interface/SummaryGenerator.h
   @class SummaryGenerator
   @author M.Wingham, R.Bainbridge
   @brief : Fills summary histograms.
*/

class SummaryGenerator {

 public: 

  // ---------- General ----------

  SummaryGenerator();
  virtual ~SummaryGenerator() {;}

  // Some useful typedefs
  typedef std::pair<float,float> Data;
  typedef std::vector<Data> BinData;
  typedef std::map<std::string,BinData> HistoData;

  /** Creates instance of derived class based on view parameter. */
  static SummaryGenerator* instance( const sistrip::View& );

  // ---------- Contruct name and TObject ----------

  /** Constructs the summary histogram name. */
  static std::string name( const sistrip::Task&, 
			   const sistrip::SummaryHisto&, 
			   const sistrip::SummaryType&,
			   const sistrip::View&, 
			   const std::string& directory );

  /** Creates instance of derived class based on view parameter. */
  static TH1* histogram( const sistrip::SummaryType&,
			 const uint32_t& xbins );
  
  // ---------- Fill map and update histogram ----------

  /** Fills the map that is used to generate the histogram(s). */
  void fillMap( const std::string& top_level_dir,
		const sistrip::Granularity&,
		const uint32_t& key, 
		const float& value, 
	const float& error = 0. );


  /** Clear the map that is used to generate the histogram(s). */
  void clearMap();
  
  /** Creates simple 1D distribution of the parameter values. */
  void summaryDistr( TH1& ); 
  
  /** Creates a 1D histogram, with the weighted sum of the parameter
      (y-axis) binned as a function of position within the given
      logical structure, which is view-dependent (x-axis). */
  void summary1D( TH1& );
  
  /** Creates a profile histogram, with individual values of the
      parameter (y-axis) binned as a function of position within the
      given logical structure, which is view-dependent (x-axis). */
  void summary2D( TH1& );

  /** Creates a profile histogram, with the mean and spread of the
      parameter (y-axis) binned as a function of position within the
      given logical structure, which is view-dependent (x-axis). */
  void summaryProf( TH1& );

  // ---------- Histogram formatting ----------
  
  /** Some generic formatting of histogram. */
  void format( const sistrip::Task&, 
	       const sistrip::SummaryHisto&, 
	       const sistrip::SummaryType&,
	       const sistrip::View&, 
	       const std::string& directory,
	       const sistrip::Granularity&,
	       TH1& );
  
  /** Optionally set axis label */ 
  inline void axisLabel( const std::string& );
  
  /** Retrieve size of map (ie, number of bins). */
  inline uint32_t size() const;
  
 protected: // ---------- Protected methods and data ----------
  
  /** Fills the map used to generate the histogram. */
  virtual void fill( const std::string& top_level_dir,
		     const sistrip::Granularity&,
		     const uint32_t& key, 
		     const float& value, 
		     const float& error );
  
  /** A map designed to holds a set of values. The map containing
      these values should be indexed by a key.*/  
  HistoData map_;

  uint32_t entries_;
  float max_;
  float min_;
  std::string label_;
  
};

uint32_t SummaryGenerator::size() const { return map_.size(); }
void SummaryGenerator::axisLabel( const std::string& label ) { label_ = label; }

#endif // DQM_SiStripCommon_SummaryGenerator_H
