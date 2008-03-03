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
 
#include <vector>

class L1CaloRegion;
class L1GlobalCaloTrigger;

class gctTestHfEtSums
{
public:

  // structs and typedefs
  typedef std::vector<L1CaloRegion> RegionsVector;

  // Constructor and destructor
  gctTestHfEtSums();
  ~gctTestHfEtSums();

  /// Reset stored sums
  void reset();

  /// Read the input jet data from the jetfinders (after GCT processing).
  void fillExpectedHfSums(const RegionsVector& inputRegions);

  /// Check the Ht summing algorithms
  bool checkHfEtSums(const L1GlobalCaloTrigger* gct) const;

private:

  unsigned m_expectedRing0EtSumPositiveEta;
  unsigned m_expectedRing0EtSumNegativeEta;
  unsigned m_expectedRing1EtSumPositiveEta;
  unsigned m_expectedRing1EtSumNegativeEta;
  unsigned m_expectedTowerCountPositiveEta;
  unsigned m_expectedTowerCountNegativeEta;

};

#endif /*GCTTEST_H_*/
