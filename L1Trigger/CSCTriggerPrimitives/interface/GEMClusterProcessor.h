#ifndef L1Trigger_CSCTriggerPrimitives_GEMClusterProcessor_h
#define L1Trigger_CSCTriggerPrimitives_GEMClusterProcessor_h

/** \class GEMClusterProcessor
 *
 * \author Sven Dildick (Rice University)
 *
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
  void run(const GEMPadDigiClusterCollection*);

  /* Returns clusters for a given BX */
  std::vector<GEMInternalCluster> getClusters(int bx) const;

  /* Returns clusters around deltaBX for a given BX */
  std::vector<GEMInternalCluster> getClusters(int bx, int deltaBX) const;

  /* Returns coincidence clusters for a given BX */
  std::vector<GEMInternalCluster> getCoincidenceClusters(int bx) const;

  /** Returns vector of CoPads in the read-out time window, if any. */
  std::vector<GEMCoPadDigi> readoutCoPads() const;

  bool hasGE21Geometry16Partitions() const { return hasGE21Geometry16Partitions_; }

  void setESLookupTables(const CSCL1TPLookupTableME11ILT* conf);

  void setESLookupTables(const CSCL1TPLookupTableME21ILT* conf);

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
  const int region_;
  const int station_;
  const int chamber_;
  bool isEven_;

  unsigned int maxDeltaPad_;
  unsigned int maxDeltaBX_;
  unsigned int maxDeltaRoll_;

  bool hasGE21Geometry16Partitions_;

  // output collection
  std::vector<GEMInternalCluster> clusters_;

  const CSCL1TPLookupTableME11ILT* lookupTableME11ILT_;
  const CSCL1TPLookupTableME21ILT* lookupTableME21ILT_;
};

#endif
