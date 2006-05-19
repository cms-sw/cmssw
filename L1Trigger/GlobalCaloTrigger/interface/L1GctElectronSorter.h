#ifndef L1GCTELECTRONSORTER_H_
#define L1GCTELECTRONSORTER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>
#include <functional>
#include <ostream>

using std::binary_function;

class L1GctSourceCard;


class L1GctElectronSorter : public L1GctProcessor
{
public:
  friend std::ostream& operator<<(std::ostream& s,const L1GctElectronSorter& card);   
 ///
  /// constructor; set type (isolated or non-isolated)
  L1GctElectronSorter(int id, bool iso=true);
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
  /// set an input Source Card pointer
  void setInputSourceCard(unsigned i, L1GctSourceCard* sc);
  ///
  /// set input candidate
  void setInputEmCand(L1GctEmCand cand);
  ///	
  /// get input candidates
  inline std::vector<L1GctEmCand> InputCands() { return m_inputCands; }
  ///
  /// get output candidates
  inline std::vector<L1GctEmCand> OutputCands() { return m_outputCands; }
  ///Prints size of vectors and values of pointers and internal variables
  void print();	

private:

  // comparison operator for sort
  struct rank_gt : public binary_function<L1GctEmCand, L1GctEmCand, bool> {
    bool operator()(const L1GctEmCand& x, const L1GctEmCand& y) { return x.rank() > y.rank(); }
  };
	
private:
  ///
  /// algo ID
  int m_id;
  ///
  /// type of electrons to sort (isolated or non isolated)
  bool m_emCandType;
  ///
  /// source card input
  std::vector<L1GctSourceCard*> m_theSCs;
  ///
  /// input data
  std::vector<L1GctEmCand> m_inputCands;
  ///
  /// output data
  std::vector<L1GctEmCand> m_outputCands;
  
};


#endif /*L1GCTELECTRONSORTER_H_*/
