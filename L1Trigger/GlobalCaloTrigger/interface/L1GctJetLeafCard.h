#ifndef L1GCTJETLEAFCARD_H_
#define L1GCTJETLEAFCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinder.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

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
  L1GctJetLeafCard(int id, int iphi);
  ~L1GctJetLeafCard();

  /// clear internal buffers
  virtual void reset();

  /// set the input buffers
  virtual void fetchInput();
 
  /// process the data and set outputs
  virtual void process();

  /// add a Source Card
  void setInputSourceCard(int i, L1GctSourceCard* card);

  /// get the input data
  std::vector<L1GctRegion> getInputRegions() const;
    
  // get the jet output
  std::vector<L1GctJetCand> getOutputJetsA() const { return jetFinderA->getJets(); }  ///< Output jetfinder A jets (lowest jetFinder in phi)
  std::vector<L1GctJetCand> getOutputJetsB() const { return jetFinderB->getJets(); }  ///< Output jetfinder B jets (middle jetFinder in phi)
  std::vector<L1GctJetCand> getOutputJetsC() const { return jetFinderC->getJets(); }  ///< Ouptut jetfinder C jets (highest jetFinder in phi)
    
  /// get the Ex output
  inline L1GctEtComponent outputEx() const { return m_exSum; }
   
  /// get the Ey output
  inline L1GctEtComponent outputEy() const { return m_eySum; }
    
  /// get the Et output
  inline L1GctScalarEtVal outputEt() const { return m_etSum; }
  inline L1GctScalarEtVal outputHt() const { return m_htSum; }
   
  static const int MAX_JET_FINDERS = 3;

private:

  static const int MAX_SOURCE_CARDS = 15;

  // Leaf card ID
  int m_id;

  // internal algorithms
  L1GctJetFinder* jetFinderA;  ///< lowest jetFinder in phi
  L1GctJetFinder* jetFinderB;  ///< middle jetFinder in phi
  L1GctJetFinder* jetFinderC;  ///< highest jetFinder in phi

  // pointers to data source
  std::vector<L1GctSourceCard*> m_sourceCards;
  // internal data (other than jets)

  int phiPosition;

  L1GctEtComponent m_exSum;
  L1GctEtComponent m_eySum;
  L1GctScalarEtVal m_etSum;
  L1GctScalarEtVal m_htSum;
};

#endif /*L1GCTJETLEAFCARD_H_*/
