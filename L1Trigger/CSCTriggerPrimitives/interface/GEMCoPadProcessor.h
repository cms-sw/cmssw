#ifndef L1Trigger_CSCTriggerPrimitives_GEMCoPadProcessor_h
#define L1Trigger_CSCTriggerPrimitives_GEMCoPadProcessor_h

/** \class GEMCoPadProcessor
 *
 * \author Sven Dildick (TAMU)
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigi.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/GEMInternalCluster.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCLUTReader.h"

#include <vector>

class GEMCoPadProcessor {
public:
  /** Normal constructor. */
  GEMCoPadProcessor(int region, unsigned station, unsigned chamber, const edm::ParameterSet& conf);

  /** Clear copad vector */
  void clear();

  /** Runs the CoPad processor code. Called in normal running -- gets info from
      a collection of pad digis. */
  std::vector<GEMCoPadDigi> run(const GEMPadDigiCollection*);

  /** Runs the CoPad processor code. */
  std::vector<GEMInternalCluster> run(const GEMPadDigiClusterCollection*);

  /** Returns vector of CoPads in the read-out time window, if any. */
  const std::vector<GEMCoPadDigi>& readoutCoPads() const;

  // declusterizes the clusters into single pad digis
  void declusterize(const GEMPadDigiClusterCollection*, GEMPadDigiCollection&) const;

private:
  // put coincidence clusters in GEMInternalCluster vector
  void addCoincidenceClusters(const GEMPadDigiClusterCollection*);

  // put single clusters in GEMInternalCluster vector who are not
  // part of any coincidence cluster
  void addSingleClusters(const GEMPadDigiClusterCollection*);

  // translate the cluster central pad numbers into 1/8-strip number,
  // and roll numbers into min and max wiregroup numbers
  // for matching with CSC trigger primitives
  void doCoordinateConversion();

  /** Chamber id (trigger-type labels). */
  const int theRegion;
  const int theStation;
  const int theChamber;
  bool isEven_;

  unsigned int maxDeltaPad_;
  unsigned int maxDeltaBX_;
  unsigned int maxDeltaRoll_;

  // output collection
  std::vector<GEMCoPadDigi> gemCoPadV;
  std::vector<GEMInternalCluster> clusters_;

  // strings to paths of LUTs
  std::vector<std::string> padToHsME1aFiles_;
  std::vector<std::string> padToHsME1bFiles_;
  std::vector<std::string> padToHsME21Files_;

  std::vector<std::string> padToEsME1aFiles_;
  std::vector<std::string> padToEsME1bFiles_;
  std::vector<std::string> padToEsME21Files_;

  std::vector<std::string> rollToMaxWgME11Files_;
  std::vector<std::string> rollToMinWgME11Files_;
  std::vector<std::string> rollToMaxWgME21Files_;
  std::vector<std::string> rollToMinWgME21Files_;

  // unique pointers to the luts
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_hs_ME1a_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_hs_ME1a_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_hs_ME1b_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_hs_ME1b_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_hs_ME21_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_hs_ME21_odd_;

  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_es_ME1a_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_es_ME1a_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_es_ME1b_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_es_ME1b_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_es_ME21_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_pad_es_ME21_odd_;

  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l1_max_wg_ME11_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l1_max_wg_ME11_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l1_min_wg_ME11_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l1_min_wg_ME11_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l1_max_wg_ME21_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l1_max_wg_ME21_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l1_min_wg_ME21_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l1_min_wg_ME21_odd_;

  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l2_max_wg_ME11_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l2_max_wg_ME11_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l2_min_wg_ME11_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l2_min_wg_ME11_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l2_max_wg_ME21_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l2_max_wg_ME21_odd_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l2_min_wg_ME21_even_;
  std::unique_ptr<CSCLUTReader> GEMCSCLUT_roll_l2_min_wg_ME21_odd_;
};

#endif
