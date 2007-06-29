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

  inline const uint16_t& vpsp0() const;
  inline const uint16_t& vpsp1() const; 

  inline const Histo& hVpsp0() const;
  inline const Histo& hVpsp1() const;
  
  void print( std::stringstream&, uint32_t not_used = 0 );
  
 private:
  
  void reset();
  void extract( const std::vector<TH1*>& );
  void analyse();

 private:
  
  /** VPSP setting for APV0 */
  uint16_t vpsp0_; // 
  /** VPSP setting for APV1 */
  uint16_t vpsp1_;
  
  /** "VPSP scan" histo for APV0 */
  Histo hVpsp0_;
  /** "VPSP scan" histo for APV1 */
  Histo hVpsp1_;

 private:

  void deprecated();
  void anal(const std::vector<const TProfile*>& histos, 
	    std::vector<unsigned short>& monitorables);
  
};

const uint16_t& VpspScanAnalysis::vpsp0() const { return vpsp0_; }
const uint16_t& VpspScanAnalysis::vpsp1() const { return vpsp1_; }
const VpspScanAnalysis::Histo& VpspScanAnalysis::hVpsp0() const { return hVpsp0_; }
const VpspScanAnalysis::Histo& VpspScanAnalysis::hVpsp1() const { return hVpsp1_; }

#endif // DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H

