#ifndef L1GCTELECTRONLEAFCARD_H_
#define L1GCTELECTRONLEAFCARD_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"

#include <vector>
#include <ostream>

/*
 * \Class L1GctEmLeafCard
 * \Class does the 1st stage sorting of electron candidates.
 *
 * This class get the electron candidates from the source cards
 * and do the 1st stage sorting of them.
 *
 * \author  Jim Brooke
 * \date    20/02/06
 */

class L1GctEmLeafCard : L1GctProcessor {
 public:
  static const unsigned N_SOURCE_CARDS;
  static const unsigned N_SORTERS;

 public:
  /// construct with ID and vector of pointers to Source Cards
  L1GctEmLeafCard(int id, std::vector<L1GctSourceCard*> srcCards);
  ///
  /// destruct
  ~L1GctEmLeafCard();
  ///
  /// clear buffers
  virtual void reset();
  ///
  /// fetch input data
  virtual void fetchInput();
  ///
  /// process the event
  virtual void process();	
  ///
  /// get ID
  int id() { return m_id; }
  ///
  /// get the output candidates
  std::vector<L1GctEmCand> getOutputIsoEmCands(int fpga);
  ///
  /// get the output candidates
  std::vector<L1GctEmCand> getOutputNonIsoEmCands(int fpga);
  ///
  /// overload of cout operator
  friend std::ostream& operator<<(std::ostream& s,const L1GctEmLeafCard& card);

private:
  /// card ID (0 or 1)
  int m_id;
  ///  
  /// processing - 0,1 are iso sorters, 2,3 are non-iso
  std::vector<L1GctElectronSorter*> m_sorters;
  ///
  /// pointers to data source
  std::vector<L1GctSourceCard*> m_sourceCards;
  
};

std::ostream& operator<<(std::ostream& s,const L1GctEmLeafCard& card);
#endif /*L1GCTELECTRONLEAFCARD_H_*/
