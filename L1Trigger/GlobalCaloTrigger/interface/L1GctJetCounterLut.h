#ifndef L1GCTJETCOUNTERLUT_H_
#define L1GCTJETCOUNTERLUT_H_

#define JET_COUNTER_LUT_ADD_BITS 16

#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"

#include "L1Trigger/GlobalCaloTrigger/src/L1GctLut.h"

class L1GctJetCand;

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


class L1GctJetCounterLut : public L1GctLut<JET_COUNTER_LUT_ADD_BITS,1>

{
public:

  // Definitions.
  static const int NAddress;

  /// Construct with a list of cuts (most general case)
  L1GctJetCounterLut(const L1GctJetCounterSetup::cutsListForJetCounter& cuts);
  /// Construct with just one cut
  L1GctJetCounterLut(const L1GctJetCounterSetup::cutDescription& cut);
  /// Default constructor
  L1GctJetCounterLut();
  /// Copy constructor
  L1GctJetCounterLut(const L1GctJetCounterLut& lut);
  /// Destructor
  virtual ~L1GctJetCounterLut();
  
  /// Overload = operator
  L1GctJetCounterLut operator= (const L1GctJetCounterLut& lut);

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetCounterLut& lut);

  /// Checks whether jet passes the cut
  bool passesCut(const L1GctJetCand jet) const;
  bool passesCut(const uint16_t lutAddress) const;
  
  /// Return the number of cuts
  unsigned nCuts() const { return m_cutList.size(); }

  /// Return the cut descriptions
  L1GctJetCounterSetup::cutsListForJetCounter cutList() const { return m_cutList; }

protected:
  

  virtual uint16_t value (const uint16_t lutAddress) const;

private:

  L1GctJetCounterSetup::cutsListForJetCounter m_cutList;

  // PRIVATE MEMBER FUNCTIONS
  bool checkCut (const L1GctJetCounterSetup::cutDescription cut) const;
  bool jetPassesThisCut (const L1GctJetCand jet, const unsigned i) const;

  // locally calculated jet properties
  unsigned rctEta(const L1GctJetCand jet) const;
  unsigned globalPhi(const L1GctJetCand jet) const;
  
};


std::ostream& operator << (std::ostream& os, const L1GctJetCounterLut& lut);

#endif /*L1GCTJETCOUNTERLUT_H_*/
