#ifndef L1Trigger_CSCTriggerPrimitives_CSCGEMMatcher
#define L1Trigger_CSCTriggerPrimitives_CSCGEMMatcher

/** \class CSCGEMMatcher
 *
 * Helper class to check if an ALCT or CLCT matches with a GEMInternalCluster
 *
 * \author Sven Dildick (Rice University)
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>

class CSCLUTReader;
class CSCALCTDigi;
class CSCCLCTDigi;
class GEMInternalCluster;

class CSCGEMMatcher {
public:
  typedef std::vector<GEMInternalCluster> GEMInternalClusters;

  CSCGEMMatcher(int endcap,
                unsigned station,
                unsigned chamber,
                const edm::ParameterSet& tmbParams,
                const edm::ParameterSet& luts);

  // calculate the bending angle
  unsigned calculateGEMCSCBending(const CSCCLCTDigi& clct, const GEMInternalCluster& cluster) const;

  // match by BX

  // coincidences
  void matchingClustersBX(const CSCALCTDigi& alct,
                          const GEMInternalClusters& clusters,
                          GEMInternalClusters& selected) const;

  // coincidences
  void matchingClustersBX(const CSCCLCTDigi& clct,
                          const GEMInternalClusters& clusters,
                          GEMInternalClusters& selected) const;

  // coincidences or single clusters
  void matchingClustersBX(const CSCALCTDigi& alct,
                          const CSCCLCTDigi& clct,
                          const GEMInternalClusters& clusters,
                          GEMInternalClusters& selected) const;

  // match by location

  // coincidences
  void matchingClustersLoc(const CSCALCTDigi& alct,
                           const GEMInternalClusters& clusters,
                           GEMInternalClusters& selected) const;

  // coincidences
  void matchingClustersLoc(const CSCCLCTDigi& clct,
                           const GEMInternalClusters& clusters,
                           GEMInternalClusters& selected) const;

  // match by 1/2-strip
  bool matchedClusterLocHS(const CSCCLCTDigi& clct, const GEMInternalCluster& cluster) const;

  // match by 1/8-strip
  bool matchedClusterLocES(const CSCCLCTDigi& clct, const GEMInternalCluster& cluster) const;

  // coincidences or single clusters
  void matchingClustersLoc(const CSCALCTDigi& alct,
                           const CSCCLCTDigi& clct,
                           const GEMInternalClusters& clusters,
                           GEMInternalClusters& selected) const;

  // match by BX and location

  // coincidences
  void matchingClustersBXLoc(const CSCALCTDigi& alct,
                             const GEMInternalClusters& clusters,
                             GEMInternalClusters& selected) const;

  // coincidences
  void matchingClustersBXLoc(const CSCCLCTDigi& clct,
                             const GEMInternalClusters& clusters,
                             GEMInternalClusters& selected) const;

  // coincidences or single clusters
  void matchingClustersBXLoc(const CSCALCTDigi& alct,
                             const CSCCLCTDigi& clct,
                             const GEMInternalClusters& clusters,
                             GEMInternalClusters& selected) const;

  // best matching clusters
  void bestClusterBXLoc(const CSCALCTDigi& alct, const GEMInternalClusters& clusters, GEMInternalCluster& best) const;

  // coincidences
  void bestClusterBXLoc(const CSCCLCTDigi& clct, const GEMInternalClusters& clusters, GEMInternalCluster& best) const;

  // coincidences or single clusters
  void bestClusterBXLoc(const CSCALCTDigi& alct,
                        const CSCCLCTDigi& clct,
                        const GEMInternalClusters& clusters,
                        GEMInternalCluster& best) const;

private:
  // calculate slope correction
  int CSCGEMSlopeCorrector(const bool isFacing, const bool isL1orCopad, const int cscSlope) const;

  unsigned endcap_;
  unsigned station_;
  unsigned ring_;
  unsigned chamber_;
  bool isEven_;

  unsigned maxDeltaBXALCTGEM_;
  unsigned maxDeltaBXCLCTGEM_;

  unsigned maxDeltaHsEven_;
  unsigned maxDeltaHsOdd_;
  unsigned maxDeltaHsEvenME1a_;
  unsigned maxDeltaHsOddME1a_;

  bool assign_gem_csc_bending_;

  // strings to paths of LUTs
  std::vector<std::string> gemCscSlopeCorrectionFiles_;
  std::vector<std::string> esDiffToSlopeME1aFiles_;
  std::vector<std::string> esDiffToSlopeME1bFiles_;
  std::vector<std::string> esDiffToSlopeME21Files_;

  // unique pointers to the luts
  std::unique_ptr<CSCLUTReader> gem_csc_slope_corr_L1_ME11_even_;
  std::unique_ptr<CSCLUTReader> gem_csc_slope_corr_L2_ME11_even_;
  std::unique_ptr<CSCLUTReader> gem_csc_slope_corr_L1_ME11_odd_;
  std::unique_ptr<CSCLUTReader> gem_csc_slope_corr_L2_ME11_odd_;

  std::unique_ptr<CSCLUTReader> es_diff_slope_L1_ME1b_even_;
  std::unique_ptr<CSCLUTReader> es_diff_slope_L2_ME1b_even_;

  std::unique_ptr<CSCLUTReader> es_diff_slope_L1_ME1b_odd_;
  std::unique_ptr<CSCLUTReader> es_diff_slope_L2_ME1b_odd_;

  std::unique_ptr<CSCLUTReader> es_diff_slope_L1_ME1a_even_;
  std::unique_ptr<CSCLUTReader> es_diff_slope_L2_ME1a_even_;

  std::unique_ptr<CSCLUTReader> es_diff_slope_L1_ME1a_odd_;
  std::unique_ptr<CSCLUTReader> es_diff_slope_L2_ME1a_odd_;

  std::unique_ptr<CSCLUTReader> es_diff_slope_L1_ME21_even_;
  std::unique_ptr<CSCLUTReader> es_diff_slope_L2_ME21_even_;

  std::unique_ptr<CSCLUTReader> es_diff_slope_L1_ME21_odd_;
  std::unique_ptr<CSCLUTReader> es_diff_slope_L2_ME21_odd_;
};

#endif
