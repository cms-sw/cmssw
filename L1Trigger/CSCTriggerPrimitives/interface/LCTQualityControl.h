#ifndef L1Trigger_CSCTriggerPrimitives_LCTQualityControl_h
#define L1Trigger_CSCTriggerPrimitives_LCTQualityControl_h

/** \class
 *
 * This class checks if ALCT, CLCT and LCT products are valid
 *
 * Author: Sven Dildick
 *
 */

#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCBaseboard.h"

#include <vector>
#include <algorithm>

class LCTQualityControl : public CSCBaseboard {
public:
  // constructor
  LCTQualityControl(unsigned endcap,
                    unsigned station,
                    unsigned sector,
                    unsigned subsector,
                    unsigned chamber,
                    const edm::ParameterSet& conf);

  /** Default destructor. */
  ~LCTQualityControl() override = default;

  // Check if the ALCT is valid
  void checkValidReadout(const CSCALCTDigi& alct) const;
  void checkValid(const CSCALCTDigi& alct, unsigned max_stubs = CSCConstants::MAX_ALCTS_PER_PROCESSOR) const;

  // Check if the CLCT is valid
  void checkValid(const CSCCLCTDigi& lct, unsigned max_stubs = CSCConstants::MAX_CLCTS_PER_PROCESSOR) const;

  // Check if the LCT is valid - TMB version
  void checkValid(const CSCCorrelatedLCTDigi& lct) const;

  // Check if the LCT is valid - MPC version
  void checkValid(const CSCCorrelatedLCTDigi& lct, const unsigned station, const unsigned ring) const;

  void checkRange(int parameter, int min_value, int max_value, const std::string& comment, unsigned& errors) const;

  template <class T>
  void reportErrors(const T& lct, const unsigned errors) const;

  // no more than 2 LCTs per BX in the readout
  void checkMultiplicityBX(const std::vector<CSCALCTDigi>& alcts) const;
  void checkMultiplicityBX(const std::vector<CSCCLCTDigi>& clcts) const;
  void checkMultiplicityBX(const std::vector<CSCCorrelatedLCTDigi>& lcts) const;
  template <class T>
  void checkMultiplicityBX(const std::vector<T>& lcts, unsigned nLCT) const;

  // for Phase-1 patterns
  int getSlopePhase1(int pattern) const;

  // CSC max strip & max wire
  unsigned get_csc_max_wire(int station, int ring) const;
  unsigned get_csc_max_halfstrip(int station, int ring) const;
  unsigned get_csc_max_quartstrip(int station, int ring) const;
  unsigned get_csc_max_eightstrip(int station, int ring) const;

  // slope values
  std::pair<int, int> get_csc_clct_min_max_slope() const;

  // CLCT min, max CFEB numbers
  std::pair<unsigned, unsigned> get_csc_min_max_cfeb(int station, int ring) const;

  // CSC min, max pattern
  std::pair<unsigned, unsigned> get_csc_min_max_pattern(bool isRun3) const;
  std::pair<unsigned, unsigned> get_csc_lct_min_max_pattern() const;

  // CSC max quality
  unsigned get_csc_alct_max_quality(int station, int ring, bool runGEMCSC) const;
  unsigned get_csc_clct_max_quality() const;
  unsigned get_csc_lct_max_quality() const;

private:
  // min number of layers for a CLCT
  unsigned nplanes_clct_hit_pattern;
};

template <class T>
void LCTQualityControl::checkMultiplicityBX(const std::vector<T>& collection, unsigned nLCT) const {
  std::unordered_map<int, unsigned> freq;
  // check each BX
  for (const auto& p : collection) {
    if (p.isValid()) {
      freq[p.getBX()]++;

      // too many ALCTs, CLCTs or LCTs in this BX
      if (freq[p.getBX()] > nLCT) {
        edm::LogError("LCTQualityControl") << "Collection with more than " << nLCT << " in BX " << p.getBX();
        break;
      }
    }
  }
}

#endif
