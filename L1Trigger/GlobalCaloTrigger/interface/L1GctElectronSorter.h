#ifndef L1GCTELECTRONSORTER_H_
#define L1GCTELECTRONSORTER_H_

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>
#include <functional>
#include <ostream>


/*!
 * \Class L1GctElectronSorter
 * \brief Class that sorts electron candidates
 *
 * This class can be constructed to sort iso or non-iso
 * electron candidates.
 * The electrons are sorted in ascending order and the 4
 * highest in rank will be returned.
 * It represents the 1st stage sorter FPGA's on the electron
 * leaf cards.
 *
 * \author  Maria Hansen
 * \date    21/04/06
 */

class L1GctElectronSorter : public L1GctProcessor
{
 public:
  
  /// constructor; set type (isolated or non-isolated)
  L1GctElectronSorter(int nInputs, bool iso);
  ///   
  ~L1GctElectronSorter();
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
  /// set input candidate
  void setInputEmCand(L1CaloEmCand cand);
  ///	
  /// get input candidates
  inline std::vector<L1CaloEmCand> getInputCands() { return m_inputCands; }
  ///
  /// get output candidates
  inline std::vector<L1GctEmCand> getOutputCands() { return m_outputCands; }
  ///
  /// overload of cout operator
  friend std::ostream& operator<<(std::ostream& s,const L1GctElectronSorter& card);  
 private:
  /// comparison operator for sort
  struct rank_gt : public std::binary_function<L1GctEmCand, L1GctEmCand, bool> {
    bool operator()(const L1GctEmCand& x, const L1GctEmCand& y) {
      if(x.rank()!=y.rank()){return x.rank() > y.rank();
      }else{if(x.etaIndex()!=y.etaIndex()){return y.etaIndex() > x.etaIndex();
	}else{ return x.phiIndex() > y.phiIndex();}}}};

  /// converts from L1CaloEmCand to L1GctEmCand
  //  std::vector<L1GctEmCand> convertCaloToGct(std::vector<L1CaloEmCand> cand);
  
 private:
  ///
  /// algo ID (is it FPGA 1 or 2 processing)
  int m_id;
  ///
  /// type of electron to sort (isolated = 0 or non isolated = 1)
  bool m_isolation;
  ///
  /// input data
  std::vector<L1CaloEmCand> m_inputCands;
  ///
  /// output data
  std::vector<L1GctEmCand> m_outputCands;
  
};

std::ostream& operator<<(std::ostream& s,const L1GctElectronSorter& card); 
#endif /*L1GCTELECTRONSORTER_H_*/
