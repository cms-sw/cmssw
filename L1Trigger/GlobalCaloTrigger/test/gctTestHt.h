#ifndef GCTTESTHT_H_
#define GCTTESTHT_H_

/*!
 * \class gctTestHt
 * \brief Test of the Ht and jet counts
 * 
 * Ht and jet count sum test functionality migrated from standalone test programs
 * March 2009 - removed all references to jet counts, and renamed gctTestHt
 *
 * \author Greg Heath
 * \date March 2007
 *
 */

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"

#include <vector>

class L1GlobalCaloTrigger;
class L1CaloEtScale;
class L1GctJetFinderParams;
class L1GctJetLeafCard;
class L1GctJetFinderBase;

class gctTestHt {
public:
  // structs and typedefs
  typedef std::vector<L1GctJetCand> JetsVector;
  typedef std::vector<L1GctJet> RawJetsVector;

  typedef L1GctJet::lutPtr lutPtr;
  typedef std::vector<lutPtr> lutPtrVector;

  struct rawJetData {
    RawJetsVector jets;
    unsigned httSum;
    int htxSum;
    int htySum;
    bool httOverFlow;
    bool htmOverFlow;

    rawJetData() : jets(), httSum(0), htxSum(0), htySum(0), httOverFlow(false), htmOverFlow(false) {}
    rawJetData(
        const RawJetsVector jv, const unsigned htt, const int htx, const int hty, const bool httof, const bool htmof)
        : jets(jv), httSum(htt), htxSum(htx), htySum(hty), httOverFlow(httof), htmOverFlow(htmof) {}
  };

  // Constructor and destructor
  gctTestHt();
  ~gctTestHt();

  /// Configuration method
  void configure(const L1CaloEtScale* jetScale, const L1CaloEtScale* mhtScale, const L1GctJetFinderParams* jfPars);
  bool setupOk() const;

  /// Set array sizes for the number of bunch crossings
  void setBxRange(const int bxStart, const int numOfBx);

  /// Read the input jet data from the jetfinders (after GCT processing).
  void fillRawJetData(const L1GlobalCaloTrigger* gct);

  /// Check the Ht summing algorithms
  bool checkHtSums(const L1GlobalCaloTrigger* gct) const;

private:
  //

  rawJetData rawJetFinderOutput(const L1GctJetFinderBase* jf, const unsigned phiPos, const int bx) const;

  int m_bxStart;
  int m_numOfBx;

  std::vector<rawJetData> minusWheelJetDta;
  std::vector<rawJetData> plusWheelJetData;

  const L1CaloEtScale* m_jetEtScale;
  const L1CaloEtScale* m_htMissScale;
  const L1GctJetFinderParams* m_jfPars;

  int htComponent(const unsigned Emag0, const unsigned fact0, const unsigned Emag1, const unsigned fact1) const;

  double htComponentGeVForHtMiss(int inputComponent) const;
};

#endif /*GCTTEST_H_*/
