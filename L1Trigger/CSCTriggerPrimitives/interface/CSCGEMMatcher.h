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

  void setESLookupTables(const CSCL1TPLookupTableME11ILT* conf);
  void setESLookupTables(const CSCL1TPLookupTableME21ILT* conf);

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
  // access to lookup tables via eventsetup
  const CSCL1TPLookupTableME11ILT* lookupTableME11ILT_;
  const CSCL1TPLookupTableME21ILT* lookupTableME21ILT_;

  //mitigate slope by consistency of slope indicator, if necessary
  uint16_t mitigatedSlopeByConsistency(const CSCCLCTDigi& clct) const;

  // calculate slope correction
  int CSCGEMSlopeCorrector(const bool isL1orCopad, const int cscSlope) const;

  unsigned endcap_;
  unsigned station_;
  unsigned ring_;
  unsigned chamber_;
  bool isEven_;

  unsigned maxDeltaBXALCTGEM_;
  unsigned maxDeltaBXCLCTGEM_;

  bool matchWithHS_;

  unsigned maxDeltaHsEven_;
  unsigned maxDeltaHsOdd_;
  unsigned maxDeltaHsEvenME1a_;
  unsigned maxDeltaHsOddME1a_;

  bool assign_gem_csc_bending_;
  bool mitigateSlopeByCosi_;
};

#endif
