#ifndef GCTTESTHFETSUMS_H_
#define GCTTESTHFETSUMS_H_

/*!
 * \class gctTestHfEtSums
 * \brief Test of the Hf Et sums
 * 
 * Adding tests of HF inner ring Et summing
 * and counting of fineGrain bits to the test suite
 *
 * \author Greg Heath
 * \date January 2008
 *
 */
 
#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include <vector>

class L1GlobalCaloTrigger;
class L1GctJetLeafCard;
class L1GctJetFinderBase;

class gctTestHfEtSums
{
public:

  // structs and typedefs
  typedef std::vector<L1CaloRegion> RegionsVector;

  // Constructor and destructor
  gctTestHfEtSums();
  ~gctTestHfEtSums();

  /// Read the input jet data from the jetfinders (after GCT processing).
  void fillRawJetData(const L1GlobalCaloTrigger* gct);

  /// Check the Ht summing algorithms
  bool checkHtSums(const L1GlobalCaloTrigger* gct) const;

  /// Check the jet counting algorithms
  bool checkJetCounts(const L1GlobalCaloTrigger* gct) const;

private:


};

#endif /*GCTTEST_H_*/
