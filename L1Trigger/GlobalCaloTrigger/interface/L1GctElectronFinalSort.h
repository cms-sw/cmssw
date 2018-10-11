#ifndef L1GCTELECTRONFINALSORT_H_
#define L1GCTELECTRONFINALSORT_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"

#include <vector>

/*!
 * \Class L1GctElectronFinalSort
 * \brief Does the final sorting of electron candidates
 *
 * This class can be constructed to sort iso or non-iso
 * electron candidates, which have been through the
 * 1st stage sorters on the 2 electron leaf cards.
 * The electrons are sorted in ascending order and the 4
 * highest in rank will be returned.
 * It represents the final-sorter FPGA on the concentrator card.
 *
 * \author  Maria Hansen
 * \date    12/05/06
 */


class L1GctEmLeafCard;

class L1GctElectronFinalSort : public L1GctProcessor
{
public:
  /// Use some definitions from the ElectronSorter in the leaf cards
  typedef L1GctElectronSorter::prioritisedEmCand prioritisedEmCand;
  ///     
  /// constructor
  L1GctElectronFinalSort(bool iso, L1GctEmLeafCard* posEtaCard,
                                   L1GctEmLeafCard* negEtaCard);
  ///
  /// destructor
  ~L1GctElectronFinalSort() override;
  ///
  /// get input data from sources
  void fetchInput() override;
  ///
  /// process the data, fill output buffers
  void process() override;
  ///
  /// set input data
  void setInputEmCand(unsigned i, const L1GctEmCand& cand);
  ///
  /// return input data
  inline std::vector<L1GctEmCand> getInputCands()  const { return m_inputCands; }
  ///
  /// return output data
  inline std::vector<L1GctEmCand> getOutputCands() const { return m_outputCands.contents; }
  ///
  /// overload of cout operator
  friend std::ostream& operator<<(std::ostream& s,const L1GctElectronFinalSort& cand); 
  ///
  /// check setup
  bool setupOk() const { return m_setupOk; }
  
 protected:

  /// Separate reset methods for the processor itself and any data stored in pipelines
  void resetProcessor() override;
  void resetPipelines() override;

  /// Initialise inputs with null objects for the correct bunch crossing if required
  void setupObjects() override {}

 private:
  ///
  /// type of electron candidate (iso(0) or non-iso(1))
  bool m_emCandsType;
  ///
  /// the 1st stage electron sorters
  L1GctEmLeafCard* m_thePosEtaLeafCard;
  L1GctEmLeafCard* m_theNegEtaLeafCard;
  ///
  /// input data
  std::vector<L1GctEmCand> m_inputCands;
  ///
  /// output data
  Pipeline<L1GctEmCand> m_outputCands;
  
  /// Check the setup
  bool m_setupOk;
};

std::ostream& operator<<(std::ostream& s,const L1GctElectronFinalSort& cand); 

#endif /*L1GCTELECTRONFINALSORT_H_*/
