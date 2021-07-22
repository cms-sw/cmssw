#ifndef L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboard_h
#define L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboard_h

/** \class CSCGEMMotherboard
 *
 * Class for TMBs for the GEM-CSC integrated local trigger. Inherits
 * from CSCMotherboard. Provides functionality to match
 * ALCT/CLCT to GEM single clusters or coincidences of clusters
 *
 * \author Sven Dildick (Rice University)
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/GEMClusterProcessor.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMatcher.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"

class CSCGEMMotherboard : public CSCMotherboard {
public:
  typedef std::vector<GEMInternalCluster> GEMInternalClusters;

  // standard constructor
  CSCGEMMotherboard(unsigned endcap,
                    unsigned station,
                    unsigned sector,
                    unsigned subsector,
                    unsigned chamber,
                    const edm::ParameterSet& conf);

  ~CSCGEMMotherboard() override;

  // clear stored pads and copads
  void clear();

  /*
  Use ALCTs, CLCTs, GEMs to build LCTs. Matches are attempted in the following order:
    1) ALCT-CLCT-2GEM (coincidence pad)
    2) ALCT-CLCT-GEM
    3) ALCT-CLCT
    4) CLCT-2GEM (requires CLCT with at least 4 layers)
    5) ALCT-2GEM (requires ALCT with at least 4 layers)
    Sort LCTs according to the cross-bunch-crossing algorithm, and send out best 2 LCTs
  */
  void run(const CSCWireDigiCollection* wiredc,
           const CSCComparatorDigiCollection* compdc,
           const GEMPadDigiClusterCollection* gemPads);

  /* GEM cluster processor */
  std::shared_ptr<GEMClusterProcessor> clusterProc() const { return clusterProc_; }

  // set CSC and GEM geometries for the matching needs
  void setGEMGeometry(const GEMGeometry* g) { gem_g = g; }

private:
  // match ALCT-CLCT-GEM pairs
  void matchALCTCLCTGEM(bool bunch_crossing_mask[CSCConstants::MAX_ALCT_TBINS]);

  // match CLCT-2GEM pairs. The GEM coincidence cluster BX is considered the
  // reference
  void matchCLCT2GEM(bool bunch_crossing_mask[CSCConstants::MAX_ALCT_TBINS]);

  // match ALCT-2GEM pairs. The GEM coincidence cluster BX is considered the
  // reference
  void matchALCT2GEM(bool bunch_crossing_mask[CSCConstants::MAX_ALCT_TBINS]);

  /* correlate a pair of ALCTs and a pair of CLCTs with matched clusters or coclusters
     the output is up to two LCTs */
  void correlateLCTsGEM(const CSCALCTDigi& bestALCT,
                        const CSCALCTDigi& secondALCT,
                        const CSCCLCTDigi& bestCLCT,
                        const CSCCLCTDigi& secondCLCT,
                        const GEMInternalClusters& clusters,
                        CSCCorrelatedLCTDigi& lct1,
                        CSCCorrelatedLCTDigi& lct2) const;

  /* correlate a pair of CLCTs with matched clusters or coclusters
     the output is up to two LCTs */
  void correlateLCTsGEM(const CSCCLCTDigi& bestCLCT,
                        const CSCCLCTDigi& secondCLCT,
                        const GEMInternalClusters& clusters,
                        CSCCorrelatedLCTDigi& lct1,
                        CSCCorrelatedLCTDigi& lct2) const;

  /* correlate a pair of ALCTs with matched clusters or coclusters
     the output is up to two LCTs */
  void correlateLCTsGEM(const CSCALCTDigi& bestALCT,
                        const CSCALCTDigi& secondALCT,
                        const GEMInternalClusters& clusters,
                        CSCCorrelatedLCTDigi& lct1,
                        CSCCorrelatedLCTDigi& lct2) const;

  /* Construct LCT from CSC and GEM information. Options are ALCT-CLCT-GEM, ALCT-CLCT-2GEM */
  void constructLCTsGEM(const CSCALCTDigi& alct,
                        const CSCCLCTDigi& clct,
                        const GEMInternalCluster& gem,
                        CSCCorrelatedLCTDigi& lct) const;

  /* Construct LCT from CSC and GEM information. Options are CLCT-2GEM */
  void constructLCTsGEM(const CSCCLCTDigi& clct,
                        const GEMInternalCluster& gem,
                        int trackNumber,
                        CSCCorrelatedLCTDigi& lct) const;

  /* Construct LCT from CSC and GEM information. Options are ALCT-2GEM */
  void constructLCTsGEM(const CSCALCTDigi& alct,
                        const GEMInternalCluster& gem,
                        int trackNumber,
                        CSCCorrelatedLCTDigi& lct) const;

  // helper functions to drop low quality ALCTs or CLCTs
  // without matching LCTs
  void dropLowQualityALCTNoClusters(CSCALCTDigi& alct, const GEMInternalCluster& cluster) const;
  void dropLowQualityCLCTNoClusters(CSCCLCTDigi& clct, const GEMInternalCluster& cluster) const;

  /*
    - For Run-2 GEM-CSC trigger primitives, which we temporarily have
    to integrate with the Run-2 EMTF during LS2, we sort by quality.
    Larger quality means smaller bending

    - For Run-3 GEM-CSC trigger primitives, which we have
    to integrate with the Run-3 EMTF, we sort by slope.
    Smaller slope means smaller bending
  */
  void sortLCTsByBending(std::vector<CSCCorrelatedLCTDigi>& lcts) const;

  /** Chamber id (trigger-type labels). */
  unsigned gemId;
  const GEMGeometry* gem_g;

  // map of BX to vectors of GEM clusters. Makes it easier to match objects
  std::map<int, GEMInternalClusters> clusters_;

  std::unique_ptr<CSCGEMMatcher> cscGEMMatcher_;

  /* GEM cluster processor */
  std::shared_ptr<GEMClusterProcessor> clusterProc_;

  // Drop low quality stubs in ME1/b or ME2/1
  bool drop_low_quality_alct_no_gems_;
  bool drop_low_quality_clct_no_gems_;

  // build LCT from ALCT/CLCT and GEM in ME1/b or ME2/1
  bool build_lct_from_alct_gem_;
  bool build_lct_from_clct_gem_;

  // Drop low quality stubs in ME1/a
  bool drop_low_quality_clct_no_gems_me1a_;

  // build LCT from CLCT and GEM in ME1/a
  bool build_lct_from_clct_gem_me1a_;

  // bunch crossing window cuts
  unsigned max_delta_bx_alct_gem_;
  unsigned max_delta_bx_clct_gem_;

  // assign GEM-CSC bending angle
  bool assign_gem_csc_bending_;

  bool drop_used_gems_;
  bool match_earliest_gem_only_;

  // The GE2/1 geometry should have 16 eta partitions
  // The 8-eta partition case (older prototype versions) is not supported
  bool hasGE21Geometry16Partitions_;
};

#endif
