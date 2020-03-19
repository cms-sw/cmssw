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

class L1GctEmLeafCard : public L1GctProcessor {
public:
  static const unsigned N_SORTERS;

public:
  /// construct with ID
  L1GctEmLeafCard(int id);
  ///
  /// destruct
  ~L1GctEmLeafCard() override;
  ///
  /// clear internal trigger data buffers
  void reset();
  ///
  /// fetch input data
  void fetchInput() override;
  ///
  /// process the event
  void process() override;
  ///
  /// define the bunch crossing range to process
  void setBxRange(const int firstBx, const int numberOfBx);
  ///
  /// clear input data buffers and process a new bunch crossing
  void setNextBx(const int bxnum);
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
  friend std::ostream& operator<<(std::ostream& s, const L1GctEmLeafCard& card);

  L1GctElectronSorter* getIsoElectronSorterU1() { return m_sorters.at(0); }
  L1GctElectronSorter* getNonIsoElectronSorterU1() { return m_sorters.at(1); }
  L1GctElectronSorter* getIsoElectronSorterU2() { return m_sorters.at(2); }
  L1GctElectronSorter* getNonIsoElectronSorterU2() { return m_sorters.at(3); }

protected:
  /// Separate reset methods for the processor itself and any data stored in pipelines
  void resetProcessor() override {}
  void resetPipelines() override {}

  /// Initialise inputs with null objects for the correct bunch crossing if required
  void setupObjects() override {}

private:
  /// card ID (0 or 1)
  int m_id;
  ///
  /// processing - 0,2 are iso sorters, 1,3 are non-iso
  std::vector<L1GctElectronSorter*> m_sorters;
};

std::ostream& operator<<(std::ostream& s, const L1GctEmLeafCard& card);
#endif /*L1GCTELECTRONLEAFCARD_H_*/
