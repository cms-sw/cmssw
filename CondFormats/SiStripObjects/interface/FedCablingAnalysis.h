#ifndef CondFormats_SiStripObjects_FedCablingAnalysis_H
#define CondFormats_SiStripObjects_FedCablingAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <map>
#include <cstdint>

/** 
   @class FedCablingAnalysis
   @author R.Bainbridge
   @brief Histogram-based analysis for connection loop.
*/
class FedCablingAnalysis : public CommissioningAnalysis {
public:
  // ---------- con(de)structors and typedefs ----------

  FedCablingAnalysis(const uint32_t& key);

  FedCablingAnalysis();

  ~FedCablingAnalysis() override { ; }

  typedef std::map<uint32_t, uint16_t> Candidates;

  friend class FedCablingAlgorithm;

  // ---------- public interface ----------

  /** Identifies if analysis is valid or not. */
  bool isValid() const override;

  /** FED id. */
  inline const uint16_t& fedId() const;

  /** FED channel. */
  inline const uint16_t& fedCh() const;

  /** Light level [ADC]. */
  inline const float& adcLevel() const;

  /** Container for candidate connections. */
  inline const Candidates& candidates() const;

  // ---------- misc ----------

  /** Prints analysis results. */
  void print(std::stringstream&, uint32_t not_used = 0) override;

  /** Resets analysis member data. */
  void reset() override;

  // ---------- public static data ----------

public:
  /** Threshold to identify candidate connections. */
  static const float threshold_;

  // ---------- private member data ----------

private:
  /** FED id */
  uint16_t fedId_;

  /** FED channel */
  uint16_t fedCh_;

  /** Light level [ADC]. */
  float adcLevel_;

  /** Container for candidate connections. */
  Candidates candidates_;
};

// ---------- Inline methods ----------

const uint16_t& FedCablingAnalysis::fedId() const { return fedId_; }
const uint16_t& FedCablingAnalysis::fedCh() const { return fedCh_; }
const float& FedCablingAnalysis::adcLevel() const { return adcLevel_; }
const FedCablingAnalysis::Candidates& FedCablingAnalysis::candidates() const { return candidates_; }

#endif  // CondFormats_SiStripObjects_FedCablingAnalysis_H
