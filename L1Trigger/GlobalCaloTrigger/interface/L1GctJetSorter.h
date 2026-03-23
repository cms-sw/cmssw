#ifndef L1GCTJETSORTER_H_
#define L1GCTJETSORTER_H_

#include <vector>
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCandFwd.h"

class L1GctJetSorter {
public:
  //Typedefs
  typedef std::vector<L1GctJetCand> JetVector;

  L1GctJetSorter();
  L1GctJetSorter(JetVector& inputJets);

  ~L1GctJetSorter();

  void setJets(JetVector& inputJets);

  JetVector getSortedJets() const;
  JetVector getInputJets() const;

private:
  JetVector m_inputJets;
};

#endif
