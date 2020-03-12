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

#include <vector>

class GEMCoPadProcessor {
public:
  /** Normal constructor. */
  GEMCoPadProcessor(unsigned region, unsigned station, unsigned chamber, const edm::ParameterSet& copad);

  /** Default constructor. Used for testing. */
  GEMCoPadProcessor();

  /** Clear copad vector */
  void clear();

  /** Runs the CoPad processor code. Called in normal running -- gets info from
      a collection of pad digis. */
  std::vector<GEMCoPadDigi> run(const GEMPadDigiCollection*);

  /** Runs the CoPad processor code. Called in normal running -- gets info from
      a collection of pad digi clusters. */
  std::vector<GEMCoPadDigi> run(const GEMPadDigiClusterCollection*);

  /** Maximum number of time bins. */
  enum { MAX_CoPad_BINS = 3 };

  /** Returns vector of CoPads in the read-out time window, if any. */
  const std::vector<GEMCoPadDigi>& readoutCoPads() const;

  // declusterizes the clusters into single pad digis
  void declusterize(const GEMPadDigiClusterCollection*, GEMPadDigiCollection&) const;

private:
  /** Chamber id (trigger-type labels). */
  const int theRegion;
  const int theStation;
  const int theChamber;

  /** Verbosity level: 0: no print (default).
   *                   1: print only CoPads found.
   *                   2: info at every step of the algorithm.
   *                   3: add special-purpose prints. */
  unsigned int infoV;
  unsigned int maxDeltaPad_;
  unsigned int maxDeltaBX_;
  unsigned int maxDeltaRoll_;

  // output collection
  std::vector<GEMCoPadDigi> gemCoPadV;
};

#endif
