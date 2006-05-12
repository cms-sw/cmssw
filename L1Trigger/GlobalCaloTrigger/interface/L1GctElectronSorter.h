#ifndef L1GCTELECTRONSORTER_H_
#define L1GCTELECTRONSORTER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>
#include <functional>

using std::vector;
using std::binary_function;

class L1GctSourceCard;


class L1GctElectronSorter : public L1GctProcessor
{
public:
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
  inline vector<L1GctEmCand> getInputCands() { return inputCands; }
  ///
  /// get output candidates
  inline vector<L1GctEmCand> getOutputCands() { return outputCands; }
	
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
  /// type of sorter (isolated or non isolated)
  bool getIsoEmCands;
  ///
  /// source card input
  vector<L1GctSourceCard*> theSCs;
  ///
  /// input data
  vector<L1GctEmCand> inputCands;
  ///
  /// output data
  vector<L1GctEmCand> outputCands;
  
};

#endif /*L1GCTELECTRONSORTER_H_*/
