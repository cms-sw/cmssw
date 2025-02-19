//author:  Francesco Costanza (DESY)
//date:    05/05/11

#ifndef HT_H
#define HT_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"

#include "TVector2.h"
#include <vector>
#include <iostream>

template<class T>
class HT {

private:
  double Hx,Hy;

 public:
  int njet;
  TVector2 v;
  double ScalarSum;

  typedef typename edm::Handle< T > Handler;
  HT( Handler jetcoll, double ptThreshold, double maxAbsEta):
    Hx(0),
    Hy(0),
    ScalarSum(0)  
  {
    typedef typename T::const_iterator Iter;
    for (Iter jet = jetcoll->begin(); jet!=jetcoll->end(); ++jet){
      if ((jet->pt()>ptThreshold) && (abs(jet->eta())<maxAbsEta)){
	njet++;
	Hx += jet->px();
	Hy += jet->py();
	ScalarSum += jet->pt();
      }
    }
    v=TVector2(Hx,Hy);
  }
};







#endif
