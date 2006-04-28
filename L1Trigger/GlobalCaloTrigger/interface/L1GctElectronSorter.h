#ifndef L1GCTELECTRONSORTER_H_
#define L1GCTELECTRONSORTER_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>

using std::vector;

class L1GctSourceCard;

///
/// Represents a GCT Electron Sort algorithm
/// author: Maria Hansen
/// date: 21/4/2006
/// version: 1.1
///

class L1GctElectronSorter : public L1GctProcessor
{
public:
  ///
  /// constructor; set tyep (isolated or non-isolated)
  L1GctElectronSorter(bool iso=true);
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
  inline vector<L1GctEmCand> getInput() { return inputCands; }
  ///
  /// get output candidates
  inline vector<L1GctEmCand> getOutput() { return outputCands; }
	
private:
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
  ///
  /// input variable to set iso or non-iso electrons
  int theInputType;
};

#endif /*L1GCTELECTRONSORTER_H_*/
