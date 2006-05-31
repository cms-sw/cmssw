#ifndef L1GCTJETLEAFCARD_H_
#define L1GCTJETLEAFCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include <vector>

/*
 * Represents a GCT Leaf Card
 * programmed for jet finding
 * author: Jim Brooke
 * date: 20/2/2006
 * 
 */

class L1GctJetLeafCard : L1GctProcessor
{
public:
  //Statics
  static const int MAX_JET_FINDERS;  ///< Number of jetfinders per jet leaf card
  static const unsigned int MAX_SOURCE_CARDS;  ///< Number of source cards required to provide input per jet leaf card

  //Construtors/destructor
  L1GctJetLeafCard(int id, int iphi, std::vector<L1GctSourceCard*> sourceCards,
                   L1GctJetEtCalibrationLut* jetEtCalLut);
                   
  ~L1GctJetLeafCard();

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetLeafCard& card);

  /// clear internal buffers
  virtual void reset();

  /// set the input buffers
  virtual void fetchInput();
 
  /// process the data and set outputs
  virtual void process();

  /// get pointers to associated source cards
  std::vector<L1GctSourceCard*> getSourceCards() const { return m_sourceCards; }

  /// get pointers to associated jetfinders
  L1GctJetFinder* getJetFinderA() const { return m_jetFinderA; }
  L1GctJetFinder* getJetFinderB() const { return m_jetFinderB; }
  L1GctJetFinder* getJetFinderC() const { return m_jetFinderC; }

  /// get the input data
  std::vector<L1GctRegion> getInputRegions() const;
    
  // get the jet output
  std::vector<L1GctJetCand> getOutputJetsA() const { return m_jetFinderA->getJets(); }  ///< Output jetfinder A jets (lowest jetFinder in phi)
  std::vector<L1GctJetCand> getOutputJetsB() const { return m_jetFinderB->getJets(); }  ///< Output jetfinder B jets (middle jetFinder in phi)
  std::vector<L1GctJetCand> getOutputJetsC() const { return m_jetFinderC->getJets(); }  ///< Ouptut jetfinder C jets (highest jetFinder in phi)
    
  /// get the Ex output
  L1GctEtComponent getOutputEx() const { return m_exSum; }
   
  /// get the Ey output
  L1GctEtComponent getOutputEy() const { return m_eySum; }
    
  /// get the Et output
  L1GctScalarEtVal getOutputEt() const { return m_etSum; }
  L1GctScalarEtVal getOutputHt() const { return m_htSum; }
   
private:

  // Leaf card ID
  int m_id;

  // internal algorithms
  L1GctJetFinder* m_jetFinderA;  ///< lowest jetFinder in phi
  L1GctJetFinder* m_jetFinderB;  ///< middle jetFinder in phi
  L1GctJetFinder* m_jetFinderC;  ///< highest jetFinder in phi

  // pointers to data source
  std::vector<L1GctSourceCard*> m_sourceCards;
  // internal data (other than jets)

  int phiPosition;

  L1GctEtComponent m_exSum;
  L1GctEtComponent m_eySum;
  L1GctScalarEtVal m_etSum;
  L1GctScalarEtVal m_htSum;

  // PRIVATE MEMBER FUNCTIONS
  // Given a strip Et sum, perform rotations by sine and cosine
  // factors to find the corresponding Ex and Ey
  L1GctEtComponent exComponent(const L1GctScalarEtVal etStrip, const unsigned jphi) const;
  L1GctEtComponent eyComponent(const L1GctScalarEtVal etStrip, const unsigned jphi) const;
  // Here is where the rotations are actually done
  L1GctEtComponent rotateEtValue(const L1GctScalarEtVal etStrip, const unsigned fact) const;

};

std::ostream& operator << (std::ostream& os, const L1GctJetLeafCard& card);

#endif /*L1GCTJETLEAFCARD_H_*/
