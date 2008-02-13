#ifndef L1GCTELECTRONFINALSORT_H_
#define L1GCTELECTRONFINALSORT_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronSorter.h"

#include <vector>
#include <functional>
#include <ostream>

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
  typedef L1GctElectronSorter::rank_gt           rank_gt;
  ///     
  /// constructor
  L1GctElectronFinalSort(bool iso, L1GctEmLeafCard* posEtaCard,
                                   L1GctEmLeafCard* negEtaCard);
  ///
  /// destrcutor
  ~L1GctElectronFinalSort();
  ///
  /// clear internal buffers
  virtual void reset();
  ///
  /// get input data from sources
  virtual void fetchInput();
  ///
  /// process the data, fill output buffers
  virtual void process();
  ///
  /// set input data
  void setInputEmCand(unsigned i, const L1GctEmCand& cand);
  ///
  /// return input data
  inline std::vector<L1GctEmCand> getInputCands()  const { return m_inputCands; }
  ///
  /// return output data
  inline std::vector<L1GctEmCand> getOutputCands() const { return m_outputCands; }
  ///
  /// overload of cout operator
  friend std::ostream& operator<<(std::ostream& s,const L1GctElectronFinalSort& cand); 
  
 //private:
 //
 // /// comparison operator for sort
 // struct rank_gt : public std::binary_function<L1GctEmCand, L1GctEmCand, bool> {
 //   bool operator()(const L1GctEmCand& x, const L1GctEmCand& y) {
 //     if(x.rank()!=y.rank()){return x.rank() > y.rank();
 //     }else{if(x.etaIndex()!=y.etaIndex()){return y.etaIndex() > x.etaIndex();
 //     }else{ return x.phiIndex() > y.phiIndex();}}}};
 //
  
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
  std::vector<L1GctEmCand> m_outputCands;

  
};

std::ostream& operator<<(std::ostream& s,const L1GctElectronFinalSort& cand); 

#endif /*L1GCTELECTRONFINALSORT_H_*/
