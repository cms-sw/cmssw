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

class L1GctJetLeafCard : public L1GctProcessor
{
public:
  //Type declaration
  enum jetFinderType { tdrJetFinder, hardwareJetFinder, nullJetFinder };

  //Statics
  static const int MAX_JET_FINDERS;  ///< Number of jetfinders per jet leaf card

  //Typedefs
  typedef L1GctUnsignedInt<L1GctInternEtSum::kTotEtOrHtNBits> etTotalType;
  typedef L1GctUnsignedInt<L1GctInternEtSum::kTotEtOrHtNBits> etHadType;

  typedef L1GctTwosComplement<  L1GctInternEtSum::kMissExOrEyNBits > etComponentType;
  typedef L1GctTwosComplement< L1GctInternHtMiss::kMissHxOrHyNBits > htComponentType;

  typedef L1GctJetFinderBase::hfTowerSumsType hfTowerSumsType;

  enum maxValues {
    etTotalMaxValue = L1GctInternEtSum::kTotEtOrHtMaxValue,
    htTotalMaxValue = L1GctInternEtSum::kTotEtOrHtMaxValue
  };

  //Construtors/destructor
  L1GctJetLeafCard(int id, int iphi, jetFinderType jfType = tdrJetFinder);
                   
  ~L1GctJetLeafCard();

  /// set pointers to neighbours - needed to complete the setup
  void setNeighbourLeafCards(const std::vector<L1GctJetLeafCard*>& neighbours);

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
    
  /// get the output Ht components
  etComponentType getOutputHx() const { return m_hxSum; }
  etComponentType getOutputHy() const { return m_hySum; }
    
  /// get the Et output
  etTotalType getOutputEt() const { return m_etSum; }
  etHadType   getOutputHt() const { return m_htSum; }

  hfTowerSumsType getOutputHfSums() const { return m_hfSums; }
   
  /// Bunch crossing history acces methods
  /// get the Ex output history
  std::vector< etComponentType > getAllOutputEx() const { return m_exSumPipe.contents; }
   
  /// get the Ey output history
  std::vector< etComponentType > getAllOutputEy() const { return m_eySumPipe.contents; }

  /// get the output Ht components history
  std::vector< htComponentType > getAllOutputHx() const { return m_hxSumPipe.contents; }
  std::vector< htComponentType > getAllOutputHy() const { return m_hySumPipe.contents; }
    
  /// get the Et output history
  std::vector< etTotalType > getAllOutputEt() const { return m_etSumPipe.contents; }
  std::vector< etHadType >   getAllOutputHt() const { return m_htSumPipe.contents; }

  std::vector< hfTowerSumsType > getAllOutputHfSums() const { return m_hfSumsPipe.contents; }
   
  /// get the Et sums in internal component format
  std::vector< L1GctInternEtSum  > getInternalEtSums() const;
  std::vector< L1GctInternHtMiss > getInternalHtMiss() const;

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
  htComponentType m_hxSum;
  htComponentType m_hySum;
  etTotalType m_etSum;
  etHadType   m_htSum;

  hfTowerSumsType m_hfSums;

  // stored copies of output data
  Pipeline<etComponentType> m_exSumPipe;
  Pipeline<etComponentType> m_eySumPipe;
  Pipeline<htComponentType> m_hxSumPipe;
  Pipeline<htComponentType> m_hySumPipe;
  Pipeline<etTotalType>     m_etSumPipe;
  Pipeline<etHadType>       m_htSumPipe;
  Pipeline<hfTowerSumsType> m_hfSumsPipe;

  bool m_ctorInputOk;

};

std::ostream& operator << (std::ostream& os, const L1GctJetLeafCard& card);

#endif /*L1GCTJETLEAFCARD_H_*/
