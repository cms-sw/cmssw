#ifndef L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboard_h
#define L1Trigger_CSCTriggerPrimitives_CSCGEMMotherboard_h

/** \class CSCGEMMotherboard
 *
 * Class for TMBs for the GEM-CSC integrated local trigger. Inherits
 * from CSCMotherboard. Provides functionality to match
 * ALCT/CLCT to GEM single clusters or coincidences of clusters
 *
 * \author Sven Dildick (Rice University)
 * \updates by Giovanni Mocellin (UC Davis)
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMotherboard.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/GEMInternalCluster.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/GEMClusterProcessor.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMatcher.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

class CSCGEMMotherboard : public CSCMotherboard {
public:
  typedef std::vector<GEMInternalCluster> GEMInternalClusters;

  // standard constructor
  CSCGEMMotherboard(unsigned endcap,
                    unsigned station,
                    unsigned sector,
                    unsigned subsector,
                    unsigned chamber,
                    CSCBaseboard::Parameters& conf);

  ~CSCGEMMotherboard() override;

  // clear stored pads and copads
  void clear();

  //helper function to convert GEM-CSC amended slopes into Run2 patterns
  uint16_t Run2PatternConverter(const int slope) const;

  struct RunContext {
    // set CSC and GEM geometries for the matching needs
    const GEMGeometry* gemGeometry_;
    const CSCGeometry* cscGeometry_;
    // access to lookup tables via eventsetup
    const CSCL1TPLookupTableCCLUT* lookupTableCCLUT_;
    const CSCL1TPLookupTableME11ILT* lookupTableME11ILT_;
    const CSCL1TPLookupTableME21ILT* lookupTableME21ILT_;
    const CSCDBL1TPParameters* parameters_;
  };

  void run(const CSCWireDigiCollection* wiredc,
           const CSCComparatorDigiCollection* compdc,
           const GEMPadDigiClusterCollection* gemPads,
           RunContext const&);

  /* GEM cluster processor */
  std::shared_ptr<GEMClusterProcessor> clusterProc() const { return clusterProc_; }

private:
  /*
  Use ALCTs, CLCTs, GEMs to build LCTs. Matches in FW are attempted in the following order:
    1) ALCT-CLCT-2GEM (coincidence pad)
    2) ALCT-CLCT-GEM
    3) ALCT-CLCT (requires CLCT with at least 4 layers)
    4) CLCT-2GEM (requires CLCT with at least 4 layers)
    5) ALCT-2GEM (requires ALCT with at least 4 layers)
    => If there are second ALCTs/CLCTs which could not be matched to GEM:
    6) Copy over valid to invalid
    7) ALCT-CLCT with unused combination
  */
  void matchALCTCLCTGEM(const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                        const CSCL1TPLookupTableME21ILT* lookupTableME21ILT);

  // correlate ALCT, CLCT with matched pads or copads
  void correlateLCTsGEM(const CSCALCTDigi& ALCT,
                        const CSCCLCTDigi& CLCT,
                        const GEMInternalClusters& clusters,
                        const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                        const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                        CSCCorrelatedLCTDigi& lct) const;

  // correlate ALCT and CLCT, no GEM
  void correlateLCTsGEM(const CSCALCTDigi& ALCT, const CSCCLCTDigi& CLCT, CSCCorrelatedLCTDigi& lct) const;

  // correlate CLCT with matched pads or copads
  void correlateLCTsGEM(const CSCCLCTDigi& CLCT,
                        const GEMInternalClusters& clusters,
                        const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                        const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                        CSCCorrelatedLCTDigi& lct) const;

  // correlate ALCT with matched pads or copads
  void correlateLCTsGEM(const CSCALCTDigi& ALCT, const GEMInternalClusters& clusters, CSCCorrelatedLCTDigi& lct) const;

  // Construct LCT from CSC and GEM information. ALCT+CLCT+GEM
  void constructLCTsGEM(const CSCALCTDigi& alct,
                        const CSCCLCTDigi& clct,
                        const GEMInternalCluster& gem,
                        const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                        const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                        CSCCorrelatedLCTDigi& lct) const;

  // Construct LCT from CSC and no GEM information. ALCT+CLCT
  void constructLCTsGEM(const CSCALCTDigi& alct, const CSCCLCTDigi& clct, CSCCorrelatedLCTDigi& lct) const;

  // Construct LCT from CSC and GEM information. CLCT+2GEM
  void constructLCTsGEM(const CSCCLCTDigi& clct,
                        const GEMInternalCluster& gem,
                        const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                        const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                        CSCCorrelatedLCTDigi& lct) const;

  // Construct LCT from CSC and GEM information. ALCT+2GEM
  void constructLCTsGEM(const CSCALCTDigi& alct, const GEMInternalCluster& gem, CSCCorrelatedLCTDigi& lct) const;

  // LCTs are sorted by quality. If there are two with the same quality,
  // then the sorting is done by the slope
  void sortLCTs(std::vector<CSCCorrelatedLCTDigi>& lcts) const;

  /** Chamber id (trigger-type labels). */
  unsigned gemId;

  // map of BX to vectors of GEM clusters. Makes it easier to match objects
  std::map<int, GEMInternalClusters> clusters_;

  /* CSCGEM matcher */
  std::unique_ptr<CSCGEMMatcher> cscGEMMatcher_;

  /* GEM cluster processor */
  std::shared_ptr<GEMClusterProcessor> clusterProc_;

  // Drop low quality stubs in ME1/b or ME2/1
  bool drop_low_quality_alct_;
  bool drop_low_quality_clct_;
  // Drop low quality stubs in ME1/a
  bool drop_low_quality_clct_me1a_;

  // build LCT from ALCT/CLCT and GEM in ME1/b or ME2/1
  bool build_lct_from_alct_gem_;
  bool build_lct_from_clct_gem_;

  // bunch crossing window cuts
  unsigned alct_gem_bx_window_size_;
  unsigned clct_gem_bx_window_size_;

  // assign GEM-CSC bending angle
  bool assign_gem_csc_bending_;

  // The GE2/1 geometry should have 16 eta partitions
  // The 8-eta partition case (older prototype versions) is not supported
  bool hasGE21Geometry16Partitions_;
};

#endif
