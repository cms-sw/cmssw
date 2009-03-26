#ifndef SimpleCutsIsolator_H
#define SimpleCutsIsolator_H

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"
#include "RecoMuon/MuonIsolation/interface/Cuts.h"


class SimpleCutsIsolator : public muonisolation::MuIsoBaseIsolator {
 public:
  SimpleCutsIsolator(const edm::ParameterSet & par):
    theCuts(par.getParameter<std::vector<double> > ("EtaBounds"),
	    par.getParameter<std::vector<double> > ("ConeSizes"),
	    par.getParameter<std::vector<double> > ("Thresholds"))
    {
    }

  virtual ResultType resultType() const {return ISOL_BOOL_TYPE;}

  virtual Result result(DepositContainer deposits) const {
    Result answer(ISOL_BOOL_TYPE);
    answer.valBool = false;
    // fail miserably...
    return answer;
  }

  virtual Result result(DepositContainer deposits, const reco::Track& tk) const {
    Result answer(ISOL_BOOL_TYPE);

    muonisolation::Cuts::CutSpec cuts_here = theCuts(tk.eta());
    
    double conesize = cuts_here.conesize;
    double dephlt = deposits.front().dep->depositWithin(conesize);
    if (dephlt<cuts_here.threshold) {
      answer.valBool = true;
    } else {
      answer.valBool = false;
    }
    return answer;
  }
  
 private:

  // Isolation cuts
  muonisolation::Cuts theCuts;


};

#endif
