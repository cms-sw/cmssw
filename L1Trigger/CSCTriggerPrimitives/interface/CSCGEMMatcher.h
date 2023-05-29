#ifndef L1Trigger_CSCTriggerPrimitives_CSCGEMMatcher
#define L1Trigger_CSCTriggerPrimitives_CSCGEMMatcher

/** \class CSCGEMMatcher
 *
 * Helper class to check if an ALCT or CLCT matches with a GEMInternalCluster
 *
 * \author Sven Dildick (Rice University)
 * \updates by Giovanni Mocellin (UC Davis)
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME11ILT.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME21ILT.h"

#include <string>
#include <vector>

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
  int calculateGEMCSCBending(const CSCCLCTDigi& clct,
                             const GEMInternalCluster& cluster,
                             const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                             const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const;

  // match by location

  // ALCT-GEM
  void matchingClustersLoc(const CSCALCTDigi& alct,
                           const GEMInternalClusters& clusters,
                           GEMInternalClusters& output) const;

  // CLCT-GEM
  void matchingClustersLoc(const CSCCLCTDigi& clct,
                           const GEMInternalClusters& clusters,
                           GEMInternalClusters& output,
                           bool ignoreALCTGEMmatch,
                           const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                           const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const;

  // matching candidate distance in 1/8 strip, always the total without extrapolation correction, if ForceTotal is true
  int matchedClusterDistES(const CSCCLCTDigi& clct,
                           const GEMInternalCluster& cluster,
                           const bool isLayer2,
                           const bool ForceTotal,
                           const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                           const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const;

  // ALCT-CLCT-GEM
  void matchingClustersLoc(const CSCALCTDigi& alct,
                           const CSCCLCTDigi& clct,
                           const GEMInternalClusters& clusters,
                           const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                           const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                           GEMInternalClusters& output) const;

  // best matching clusters by location

  // ALCT-GEM
  void bestClusterLoc(const CSCALCTDigi& alct, const GEMInternalClusters& clusters, GEMInternalCluster& best) const;

  // CLCT-GEM
  void bestClusterLoc(const CSCCLCTDigi& clct,
                      const GEMInternalClusters& clusters,
                      const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                      const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                      GEMInternalCluster& best) const;

  // ALCT-CLCT-GEM
  void bestClusterLoc(const CSCALCTDigi& alct,
                      const CSCCLCTDigi& clct,
                      const GEMInternalClusters& clusters,
                      const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                      const CSCL1TPLookupTableME21ILT* lookupTableME21ILT,
                      GEMInternalCluster& best) const;

private:
  //mitigate slope by consistency of slope indicator, if necessary
  uint16_t mitigatedSlopeByConsistency(const CSCCLCTDigi& clct,
                                       const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                                       const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const;

  // calculate slope correction
  int CSCGEMSlopeCorrector(const bool isME1a,
                           const int cscSlope,
                           bool isLayer2,
                           const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                           const CSCL1TPLookupTableME21ILT* lookupTableME21ILT) const;

  unsigned endcap_;
  unsigned station_;
  unsigned ring_;
  unsigned chamber_;
  bool isEven_;

  // enable GEM-CSC matching in ME1a and ME1b
  bool enable_match_gem_me1a_;
  bool enable_match_gem_me1b_;

  // match GEM-CSC by propagating CLCT to GEM via LUT
  bool matchCLCTpropagation_;

  // Matching interval in Half Strips (less bits to deal with in FW), but then used as Eighth Strips (es=hs*4)
  unsigned maxDeltaWG_;
  unsigned maxDeltaHsEven_;
  unsigned maxDeltaHsOdd_;

  bool assign_gem_csc_bending_;
  bool mitigateSlopeByCosi_;
};

#endif
