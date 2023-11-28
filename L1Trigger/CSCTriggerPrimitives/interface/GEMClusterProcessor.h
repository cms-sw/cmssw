#ifndef L1Trigger_CSCTriggerPrimitives_GEMClusterProcessor_h
#define L1Trigger_CSCTriggerPrimitives_GEMClusterProcessor_h

/** \class GEMClusterProcessor
 *
 * \author Sven Dildick (Rice University)
 * \updates by Giovanni Mocellin (UC Davis)
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigi.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/GEMInternalCluster.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME11ILT.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME21ILT.h"

#include <vector>

class GEMClusterProcessor {
public:
  /** Normal constructor. */
  GEMClusterProcessor(int region, unsigned station, unsigned chamber, const edm::ParameterSet& conf);

  /** Clear copad vector */
  void clear();

  /** Runs the CoPad processor code. */
  void run(const GEMPadDigiClusterCollection*,
           const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
           const CSCL1TPLookupTableME21ILT* lookupTableME21ILT);

  /* Returns clusters around deltaBX for a given BX
    The parameter option determines which clusters should be returned
    1: single and coincidence, 2: only coincidence, 3: only single
  */
  enum ClusterTypes { AllClusters = 1, SingleClusters = 2, CoincidenceClusters = 3 };
  std::vector<GEMInternalCluster> getClusters(int bx, ClusterTypes option = AllClusters) const;

  /** Returns vector of CoPads in the read-out time window, if any. */
  std::vector<GEMCoPadDigi> readoutCoPads() const;

  bool hasGE21Geometry16Partitions() const { return hasGE21Geometry16Partitions_; }

private:
  // put coincidence clusters in GEMInternalCluster vector
  void addCoincidenceClusters(const GEMPadDigiClusterCollection*);

  // put single clusters in GEMInternalCluster vector who are not
  // part of any coincidence cluster
  void addSingleClusters(const GEMPadDigiClusterCollection*);

  // translate the cluster central pad numbers into 1/8-strip number,
  // and roll numbers into min and max wiregroup numbers
  // for matching with CSC trigger primitives
  void doCoordinateConversion(const CSCL1TPLookupTableME11ILT* lookupTableME11ILT,
                              const CSCL1TPLookupTableME21ILT* lookupTableME21ILT);

  // Chamber id (trigger-type labels)
  const int region_;
  const int station_;
  const int chamber_;
  bool isEven_;

  unsigned int tmbL1aWindowSize_;
  unsigned int delayGEMinOTMB_;
  unsigned int maxDeltaPad_;
  unsigned int maxDeltaBX_;
  unsigned int maxDeltaRoll_;

  bool hasGE21Geometry16Partitions_;

  // output collection
  std::vector<GEMInternalCluster> clusters_;
};

#endif
