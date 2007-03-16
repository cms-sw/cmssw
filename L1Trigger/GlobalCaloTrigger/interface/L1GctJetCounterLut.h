#ifndef L1GCTJETCOUNTERLUT_H_
#define L1GCTJETCOUNTERLUT_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include <vector>

/*!
 * \author Greg Heath
 * \date June 2006
 */

/*! \class L1GctJetCounterLut
 * \brief Jet Counter LUT
 * 
 * Input is jet candidate
 * Output is yes/no; does the jet pass this cut?
 * 
 * Different types of cut to be set up in the constructor
 *
 * Available cut types:
 * minRank : jet passes if its Rank is at least cutValue1
 * maxRank : jet passes if its Rank is at most cutValue1
 * centralEta : jet passes if its rctEta is at most cutValue1
 * forwardEta : jet passes if its rctEta is at least cutValue1
 * phiWindow : jet passes if its globalPhi is between cutValue1 and cutValue2 (inclusive)
 *             NB if cutValue2<cutValue1 we wrap around phi=0/2pi
 * nullCutType : jet never passes
 *               (this is the default)
 *
 * If more than one cut is specified, a jet is required to pass all cuts in order to be accepted 
 *
 */


class L1GctJetCounterLut
{
public:

  // Definitions.
  // nullCutType should be the last in the list.
  enum validCutType { minRank, maxRank, centralEta, forwardEta, phiWindow, nullCutType};
  static const unsigned int MAX_CUT_TYPE;

  /// Construct with a list of cuts (most general case)
  L1GctJetCounterLut(std::vector<validCutType> cutType, std::vector<unsigned> cutValue1, std::vector<unsigned> cutValue2);
  /// Construct with just one cut (includes default constructor)
  L1GctJetCounterLut(validCutType cutType=nullCutType, unsigned cutValue1=0, unsigned cutValue2=0);
  /// Copy constructor
  L1GctJetCounterLut(const L1GctJetCounterLut& lut);
  /// Destructor
  ~L1GctJetCounterLut();
  
  /// Overload = operator
  L1GctJetCounterLut operator= (const L1GctJetCounterLut& lut);

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetCounterLut& lut);

  /// Checks whether jet passes the cut
  bool passesCut(const L1GctJetCand jet) const;
  
  /// Return the number of cuts
  unsigned nCuts() const { return m_nCuts; }

  /// Return the cut types
  std::vector<validCutType> cutType() const { return m_cutType; }

  /// Return the first cut values
  std::vector<unsigned> cutValue1() const { return m_cutValue1; }

  /// Return the second cut values
  std::vector<unsigned> cutValue2() const { return m_cutValue2; }

private:

  unsigned m_nCuts;
  std::vector<validCutType> m_cutType;
  std::vector<unsigned> m_cutValue1;
  std::vector<unsigned> m_cutValue2;

  // PRIVATE MEMBER FUNCTIONS
  void checkCut (const validCutType cutType, const unsigned cutValue1, const unsigned cutValue2) const;
  bool jetPassesThisCut (const L1GctJetCand jet, const unsigned i) const;

  // locally calculated jet properties
  unsigned rctEta(const L1GctJetCand jet) const;
  unsigned globalPhi(const L1GctJetCand jet) const;
  
};


std::ostream& operator << (std::ostream& os, const L1GctJetCounterLut& lut);

#endif /*L1GCTJETCOUNTERLUT_H_*/
