#ifndef L1GCTELECTRONLEAFCARD_H_
#define L1GCTELECTRONLEAFCARD_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"

#include <vector>
#include <ostream>

/*!
 * \Class L1GctEmLeafCard
 * \brief Emulates a leaf card programmed for electron sorting
 *
 * This class does the 1st stage sorting of the electron candidates.
 *
 * \author  Jim Brooke
 * \date    20/02/06
 */

class L1GctEmLeafCard : L1GctProcessor {
 public:
  static const unsigned N_SORTERS;

 public:
  /// construct with ID
  L1GctEmLeafCard(int id);
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

  L1GctElectronSorter* getIsoElectronSorter0()    { return m_sorters.at(0); }
  L1GctElectronSorter* getNonIsoElectronSorter0() { return m_sorters.at(1); }
  L1GctElectronSorter* getIsoElectronSorter1()    { return m_sorters.at(2); }
  L1GctElectronSorter* getNonIsoElectronSorter1() { return m_sorters.at(3); }

private:
  /// card ID (0 or 1)
  int m_id;
  ///  
  /// processing - 0,1 are iso sorters, 2,3 are non-iso
  std::vector<L1GctElectronSorter*> m_sorters;
  
};

std::ostream& operator<<(std::ostream& s,const L1GctEmLeafCard& card);
#endif /*L1GCTELECTRONLEAFCARD_H_*/
