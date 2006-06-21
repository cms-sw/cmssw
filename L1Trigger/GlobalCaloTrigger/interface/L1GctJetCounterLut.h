#ifndef L1GCTJETCOUNTERLUT_H_
#define L1GCTJETCOUNTERLUT_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

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
  /// Construct with just one cut (general case)
  L1GctJetCounterLut(validCutType cutType, unsigned cutValue1, unsigned cutValue2);
  /// Construct with just one cut and one threshold value (useful)
  L1GctJetCounterLut(validCutType cutType, unsigned cutValue1);
  /// Construct with just one cut and no values (default threshold values, ie set to zero; less useful)
  L1GctJetCounterLut(validCutType cutType);
  /// Construct null counter (reject all jets)
  L1GctJetCounterLut();
  ~L1GctJetCounterLut();
  
  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetCounterLut& lut);

  /// Checks whether jet passes the cut
  bool passesCut(const L1GctJet jet) const;
  
private:

  unsigned m_nCuts;
  std::vector<validCutType> m_cutType;
  std::vector<unsigned> m_cutValue1;
  std::vector<unsigned> m_cutValue2;

  // PRIVATE MEMBER FUNCTIONS
  void checkCut (const validCutType cutType, const unsigned cutValue1, const unsigned cutValue2) const;
  bool jetPassesThisCut (const L1GctJet jet, const unsigned i) const;
  
};


std::ostream& operator << (std::ostream& os, const L1GctJetCounterLut& lut);

#endif /*L1GCTJETCOUNTERLUT_H_*/
