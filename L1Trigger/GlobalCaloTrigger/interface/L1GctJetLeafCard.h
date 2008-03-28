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

  enum { etComponentSize=L1GctEtMiss::kEtMissNBits+2 };
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
  void reset();

  /// set the input buffers
  virtual void fetchInput();
 
  /// process the data and set outputs
  virtual void process();

  /// define the bunch crossing range to process
  void setBxRange(const int firstBx, const int numberOfBx);

  /// partially clear buffers
  void setNextBx(const int bx);

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
   
  /// Bunch crossing history acces methods
  /// get the Ex output history
  std::vector< etComponentType > getAllOutputEx() const { return m_exSumPipe.contents; }
   
  /// get the Ey output history
  std::vector< etComponentType > getAllOutputEy() const { return m_eySumPipe.contents; }
    
  /// get the Et output history
  std::vector< etTotalType > getAllOutputEt() const { return m_etSumPipe.contents; }
  std::vector< etHadType >   getAllOutputHt() const { return m_htSumPipe.contents; }

  std::vector< hfTowerSumsType > getAllOutputHfSums() const { return m_hfSumsPipe.contents; }
   
 protected:

  /// Separate reset methods for the processor itself and any data stored in pipelines
  virtual void resetProcessor();
  virtual void resetPipelines();

  /// Initialise inputs with null objects for the correct bunch crossing if required
  virtual void setupObjects() {}

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

  // stored copies of output data
  Pipeline<etComponentType> m_exSumPipe;
  Pipeline<etComponentType> m_eySumPipe;
  Pipeline<etTotalType>     m_etSumPipe;
  Pipeline<etHadType>       m_htSumPipe;
  Pipeline<hfTowerSumsType> m_hfSumsPipe;

  // PRIVATE MEMBER FUNCTIONS
  // Given a strip Et sum, perform rotations by sine and cosine
  // factors to find the corresponding Ex and Ey
  etComponentType exComponent(const etTotalType etStrip, const unsigned jphi) const;
  etComponentType eyComponent(const etTotalType etStrip, const unsigned jphi) const;
  // Here is where the rotations are actually done
  etComponentType rotateEtValue(const etTotalType etStrip, const unsigned fact) const;

};

std::ostream& operator << (std::ostream& os, const L1GctJetLeafCard& card);

#endif /*L1GCTJETLEAFCARD_H_*/
