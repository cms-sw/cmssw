#ifndef CondFormats_SiStripObjects_VpspScanAnalysis_H
#define CondFormats_SiStripObjects_VpspScanAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <cstdint>

/** 
   @class VpspScanAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Histogram-based analysis for VPSP scan.
*/
class VpspScanAnalysis : public CommissioningAnalysis {
public:
  // ---------- con(de)structors ----------

  VpspScanAnalysis(const uint32_t& key);

  VpspScanAnalysis();

  ~VpspScanAnalysis() override { ; }

  friend class VpspScanAlgorithm;

  // ---------- public interface ----------

  /** Identifies if analysis is valid or not. */
  bool isValid() const override;

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

  // ---------- misc ----------

  /** Prints analysis results. */
  void print(std::stringstream&, uint32_t not_used = 0) override;

  /** Overrides base method. */
  void summary(std::stringstream&) const override;

  /** Resets analysis member data. */
  void reset() override;

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
};

// ---------- Inline methods ----------

const VpspScanAnalysis::VInt& VpspScanAnalysis::vpsp() const { return vpsp_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::adcLevel() const { return adcLevel_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::fraction() const { return fraction_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::topEdge() const { return topEdge_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::bottomEdge() const { return bottomEdge_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::topLevel() const { return topLevel_; }
const VpspScanAnalysis::VInt& VpspScanAnalysis::bottomLevel() const { return bottomLevel_; }

#endif  // CondFormats_SiStripObjects_VpspScanAnalysis_H
