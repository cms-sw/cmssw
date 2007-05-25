#ifndef DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H
#define DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
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

  VpspScanAnalysis( const uint32_t& key );
  VpspScanAnalysis();
  virtual ~VpspScanAnalysis() {;}

  inline const VInts& vpsp() const;

  inline const Histo& hVpsp0() const;
  inline const Histo& hVpsp1() const;
  
  void print( std::stringstream&, uint32_t not_used = 0 );
  
 private:
  
  void reset();
  void extract( const std::vector<TH1*>& );
  void analyse();

 private:
  
  /** VPSP settings */
  VInts vpsp_; // 
  VInts adcLevel_;
  VInts fraction_;
  VInts topEdge_;
  VInts bottomEdge_;
  VInts topLevel_;
  VInts bottomLevel_;
  
  /** "VPSP scan" histo for APV0 */
  Histo hVpsp0_;
  /** "VPSP scan" histo for APV1 */
  Histo hVpsp1_;

 private:

  void deprecated();
  void anal(const std::vector<const TProfile*>& histos, 
	    std::vector<unsigned short>& monitorables);
  
};

const VpspScanAnalysis::VInts& VpspScanAnalysis::vpsp() const { return vpsp_; }

const VpspScanAnalysis::Histo& VpspScanAnalysis::hVpsp0() const { return hVpsp0_; }
const VpspScanAnalysis::Histo& VpspScanAnalysis::hVpsp1() const { return hVpsp1_; }

#endif // DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H

