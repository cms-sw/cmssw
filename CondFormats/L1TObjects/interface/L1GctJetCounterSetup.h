#ifndef L1GCTJETCOUNTERSETUP_H_
#define L1GCTJETCOUNTERSETUP_H_

/*!
 * \author Greg Heath
 * \date Sep 2007
 */

#include <vector>
#include <iosfwd>

/*! \class L1GctJetCounterSetup
 * \brief Jet counter setup
 * 
 *
 *============================================================================
 *
 *
 *============================================================================
 *
 */


class L1GctJetCounterSetup
{
public:

  // nullCutType should be the last in the list.
  enum validCutType { minRank, maxRank, centralEta, forwardEta, phiWindow, nullCutType};
  static const unsigned int MAX_CUT_TYPE;
  static const unsigned int MAX_JET_COUNTERS;

  struct cutDescription
  {
        validCutType cutType;
        unsigned     cutValue1;
        unsigned     cutValue2;

        cutDescription() : cutType(nullCutType), cutValue1(0), cutValue2(0) {}
  };
  typedef std::vector<cutDescription>        cutsListForJetCounter;
  typedef std::vector<cutsListForJetCounter> cutsListForWheelCard;

  L1GctJetCounterSetup(const cutsListForWheelCard&);
  L1GctJetCounterSetup();
  ~L1GctJetCounterSetup();

  unsigned numberOfJetCounters() const { return m_jetCounterCuts.size(); }

  cutsListForJetCounter getCutsForJetCounter(unsigned i) const;

  void addJetCounter(const cutsListForJetCounter& cuts);

private:

  cutsListForWheelCard m_jetCounterCuts;

};

std::ostream& operator << (std::ostream& os, const L1GctJetCounterSetup& fn);

#endif /*L1GCTJETCOUNTERSETUP_H_*/
