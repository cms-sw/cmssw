#ifndef L1GCTELECTRONLEAFCARD_H_
#define L1GCTELECTRONLEAFCARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include <vector>
#include <ostream>

///
/// Represents a GCT Leaf Card
/// programmed to sort EM candidates
/// author: Jim Brooke
/// date: 20/2/2006
/// 
///

class L1GctEmLeafCard : L1GctProcessor {
 public:
  static const unsigned N_SOURCE_CARDS;
  static const unsigned N_SORTERS;

 public:
  /// construct with ID and vector of pointers to Source Cards
  L1GctEmLeafCard(int id, std::vector<L1GctSourceCard*> srcCards);

  /// destruct
  ~L1GctEmLeafCard();
  
  /// clear buffers
  virtual void reset();
  
  /// fetch input data
  virtual void fetchInput();
  
  /// process the event
  virtual void process();	
  
  /// get the output candidates
  std::vector<L1GctEmCand> getOutputIsoEmCands(int fpga);
  
  /// get the output candidates
  std::vector<L1GctEmCand> getOutputNonIsoEmCands(int fpga);

  /// print
  friend std::ostream& operator<<(std::ostream& s,const L1GctEmLeafCard& card);

private:
  
  /// card ID
  int m_id;
  
  /// processing - 0,1 are iso sorters, 2,3 are non-iso
  std::vector<L1GctElectronSorter*> m_sorters;
  
  /// pointers to data source
  std::vector<L1GctSourceCard*> m_sourceCards;
  
};

std::ostream& operator<<(std::ostream& s,const L1GctEmLeafCard& card);
#endif /*L1GCTELECTRONLEAFCARD_H_*/
