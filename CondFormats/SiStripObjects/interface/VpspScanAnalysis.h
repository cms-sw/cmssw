#ifndef CondFormats_SiStripObjects_VpspScanAnalysis_H
#define CondFormats_SiStripObjects_VpspScanAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;
class TH1;

/** 
   @class VpspScanAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Histogram-based analysis for VPSP scan.
*/
class VpspScanAnalysis : public CommissioningAnalysis {
  
 public:

  // ---------- con(de)structors ----------

  VpspScanAnalysis( const uint32_t& key );

  VpspScanAnalysis();

  virtual ~VpspScanAnalysis() {;}

  // ---------- public interface ----------
  
  /** Identifies if analysis is valid or not. */
  bool isValid() const;

  /** VPSP settings for both APVs. */
  inline const VInt& vpsp() const;

  /** Signal levels [ADC] for VPSP settings. */
  inline const VInt& adcLevel() const;

  /** Not used. */
  inline const VInt& fraction() const;

  /** VPSP setting where baseline leaves "D1" level. */
  inline const VInt& topEdge() const;

  /** VPSP setting where baseline leaves "D0" level. */
  inline const VInt& bottomEdge() const;

  /** Signal level [ADC] for "digital one". */
  inline const VInt& topLevel() const;

  /** Signal level [ADC] for "digital zero". */
  inline const VInt& bottomLevel() const;

  /** Histogram pointer and title. */
  const Histo& histo( const uint16_t& apv ) const;

  // ---------- public print methods ----------

  /** Prints analysis results. */
  void print( std::stringstream&, uint32_t not_used = 0 );
  
  /** Overrides base method. */
  void summary( std::stringstream& ) const;

  // ---------- private methods ----------
  
 private:
  
  /** Resets analysis member data. */
  void reset();

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& );

  /** Performs histogram anaysis. */
  void analyse();

  // ---------- private member data ----------
  
 private:
  
  /** VPSP settings */
  VInt vpsp_; 

  VInt adcLevel_;

  VInt fraction_;

  VInt topEdge_;

  VInt bottomEdge_;

  VInt topLevel_;

  VInt bottomLevel_;
  
  /** Pointers and titles for histograms. */
  std::vector<Histo> histos_;

 private:

  // ---------- Private deprecated or wrapper methods ----------

  void deprecated();

  void anal(const std::vector<const TProfile*>& histos, 
	    std::vector<unsigned short>& monitorables);
  
};

// ---------- Inline methods ----------

const VpspScanAnalysis::VInt& VpspScanAnalysis::vpsp() const { return vpsp_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::adcLevel() const { return adcLevel_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::fraction() const { return fraction_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::topEdge() const { return topEdge_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::bottomEdge() const { return bottomEdge_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::topLevel() const { return topLevel_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::bottomLevel() const { return bottomLevel_; }

#endif // CondFormats_SiStripObjects_VpspScanAnalysis_H

