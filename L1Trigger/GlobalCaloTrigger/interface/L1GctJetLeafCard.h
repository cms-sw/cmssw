#ifndef L1GCTJETLEAFCARD_H_
#define L1GCTJETLEAFCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctTwosComplement.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"

#include <vector>

/*
 * \class L1GctJetLeafCard
 * \brief Emulates a leaf card programmed for jetfinding
 *
 * Represents a GCT Leaf Card
 * programmed for jet finding
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctJetLeafCard : L1GctProcessor
{
public:
  //Type declaration
  enum jetFinderType { tdrJetFinder, hardwareJetFinder };

  //Statics
  static const int MAX_JET_FINDERS;  ///< Number of jetfinders per jet leaf card

  //Construtors/destructor
  L1GctJetLeafCard(int id, int iphi, jetFinderType jfType = tdrJetFinder);
                   
  ~L1GctJetLeafCard();

  /// set pointers to neighbours - needed to complete the setup
  void setNeighbourLeafCards(std::vector<L1GctJetLeafCard*> neighbours);

  /// Check setup is Ok
  bool setupOk() const { return (m_jetFinderA->setupOk() &&
				 m_jetFinderB->setupOk() &&
				 m_jetFinderC->setupOk()); }

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetLeafCard& card);

  /// clear internal buffers
  virtual void reset();

  /// set the input buffers
  virtual void fetchInput();
 
  /// process the data and set outputs
  virtual void process();

  /// get pointers to associated jetfinders
  L1GctJetFinderBase* getJetFinderA() const { return m_jetFinderA; }
  L1GctJetFinderBase* getJetFinderB() const { return m_jetFinderB; }
  L1GctJetFinderBase* getJetFinderC() const { return m_jetFinderC; }

  // get the jet output
  std::vector<L1GctJetCand> getOutputJetsA() const { return m_jetFinderA->getJets(); }  ///< Output jetfinder A jets (lowest jetFinder in phi)
  std::vector<L1GctJetCand> getOutputJetsB() const { return m_jetFinderB->getJets(); }  ///< Output jetfinder B jets (middle jetFinder in phi)
  std::vector<L1GctJetCand> getOutputJetsC() const { return m_jetFinderC->getJets(); }  ///< Ouptut jetfinder C jets (highest jetFinder in phi)
    
  /// get the Ex output
  L1GctTwosComplement<12> getOutputEx() const { return m_exSum; }
   
  /// get the Ey output
  L1GctTwosComplement<12> getOutputEy() const { return m_eySum; }
    
  /// get the Et output
  L1GctUnsignedInt<12> getOutputEt() const { return m_etSum; }
  L1GctUnsignedInt<12> getOutputHt() const { return m_htSum; }
   
private:

  // Leaf card ID
  int m_id;

  // Which jetFinder to use?
  jetFinderType m_whichJetFinder;

  // internal algorithms
  L1GctJetFinderBase* m_jetFinderA;  ///< lowest jetFinder in phi
  L1GctJetFinderBase* m_jetFinderB;  ///< middle jetFinder in phi
  L1GctJetFinderBase* m_jetFinderC;  ///< highest jetFinder in phi
  
  // internal data (other than jets)

  int phiPosition;

  L1GctTwosComplement<12> m_exSum;
  L1GctTwosComplement<12> m_eySum;
  L1GctUnsignedInt<12> m_etSum;
  L1GctUnsignedInt<12> m_htSum;

  // PRIVATE MEMBER FUNCTIONS
  // Given a strip Et sum, perform rotations by sine and cosine
  // factors to find the corresponding Ex and Ey
  L1GctTwosComplement<12> exComponent(const L1GctUnsignedInt<12> etStrip, const unsigned jphi) const;
  L1GctTwosComplement<12> eyComponent(const L1GctUnsignedInt<12> etStrip, const unsigned jphi) const;
  // Here is where the rotations are actually done
  L1GctTwosComplement<12> rotateEtValue(const L1GctUnsignedInt<12> etStrip, const unsigned fact) const;

};

std::ostream& operator << (std::ostream& os, const L1GctJetLeafCard& card);

#endif /*L1GCTJETLEAFCARD_H_*/
