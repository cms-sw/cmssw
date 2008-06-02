#ifndef L1GCTJETLEAFCARD_H_
#define L1GCTJETLEAFCARD_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtTotal.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtHad.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtMiss.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
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

class L1GctJetCand;

class L1GctJetLeafCard : L1GctProcessor
{
public:
  //Type declaration
  enum jetFinderType { tdrJetFinder, hardwareJetFinder };

  //Statics
  static const int MAX_JET_FINDERS;  ///< Number of jetfinders per jet leaf card

  //Typedefs
  typedef L1GctUnsignedInt< L1GctEtTotal::kEtTotalNBits   > etTotalType;
  typedef L1GctUnsignedInt<   L1GctEtHad::kEtHadNBits     > etHadType;
  typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissNBits    > etMissType;
  typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissPhiNBits > etMissPhiType;

  // Use the same number of bits as the firmware (range -131072 to 131071)
  enum { etComponentSize=18 };
  typedef L1GctTwosComplement<etComponentSize> etComponentType;

  typedef L1GctJetFinderBase::hfTowerSumsType hfTowerSumsType;

  //Construtors/destructor
  L1GctJetLeafCard(int id, int iphi, jetFinderType jfType = tdrJetFinder);
                   
  ~L1GctJetLeafCard();

  /// set pointers to neighbours - needed to complete the setup
  void setNeighbourLeafCards(std::vector<L1GctJetLeafCard*> neighbours);

  /// Check setup is Ok
  bool setupOk() const;

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
  std::vector<L1GctJetCand> getOutputJetsA() const;  ///< Output jetfinder A jets (lowest jetFinder in phi)
  std::vector<L1GctJetCand> getOutputJetsB() const;  ///< Output jetfinder B jets (middle jetFinder in phi)
  std::vector<L1GctJetCand> getOutputJetsC() const;  ///< Ouptut jetfinder C jets (highest jetFinder in phi)
    
  /// get the Ex output
  etComponentType getOutputEx() const { return m_exSum; }
   
  /// get the Ey output
  etComponentType getOutputEy() const { return m_eySum; }
    
  /// get the Et output
  etTotalType getOutputEt() const { return m_etSum; }
  etHadType   getOutputHt() const { return m_htSum; }

  hfTowerSumsType getOutputHfSums() const { return m_hfSums; }
   
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

  etComponentType m_exSum;
  etComponentType m_eySum;
  etTotalType m_etSum;
  etHadType   m_htSum;

  hfTowerSumsType m_hfSums;

  // PRIVATE MEMBER FUNCTIONS
  // Given a strip Et sum, perform rotations by sine and cosine
  // factors to find the corresponding Ex and Ey
  etComponentType exComponent(const etTotalType etStrip0, const etTotalType etStrip1, const unsigned jphi) const;
  etComponentType eyComponent(const etTotalType etStrip0, const etTotalType etStrip1, const unsigned jphi) const;
  // Here is where the rotations are actually done
  etComponentType etValueForJetFinder(const etTotalType etStrip0, const unsigned fact0,
                                      const etTotalType etStrip1, const unsigned fact1) const;

};

std::ostream& operator << (std::ostream& os, const L1GctJetLeafCard& card);

#endif /*L1GCTJETLEAFCARD_H_*/
