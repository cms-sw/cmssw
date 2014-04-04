#ifndef SimpleCutsIsolator_H
#define SimpleCutsIsolator_H

#include "RecoMuon/MuonIsolation/interface/MuIsoBaseIsolator.h"
#include "RecoMuon/MuonIsolation/interface/Cuts.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"


class SimpleCutsIsolator : public muonisolation::MuIsoBaseIsolator {
 public:
 SimpleCutsIsolator(const edm::ParameterSet & par, edm::ConsumesCollector && iC):
    theCuts(par.getParameter<std::vector<double> > ("EtaBounds"),
	    par.getParameter<std::vector<double> > ("ConeSizes"),
	    par.getParameter<std::vector<double> > ("Thresholds"))
    {
    }

  virtual ResultType resultType() const {return ISOL_BOOL_TYPE;}

  virtual Result result(const DepositContainer& deposits, const edm::Event* = 0) const {
    Result answer(ISOL_BOOL_TYPE);
    answer.valBool = false;
    // fail miserably...
    return answer;
  }

  virtual Result result(const DepositContainer& deposits, const reco::Track& tk, const edm::Event* = 0) const {
    Result answer(ISOL_BOOL_TYPE);

    muonisolation::Cuts::CutSpec cuts_here = theCuts(tk.eta());
    
    double conesize = cuts_here.conesize;
    double dephlt = 0;
    unsigned int nDeps = deposits.size();
    for(unsigned int iDep = 0; iDep < nDeps; ++iDep ){
      dephlt += deposits[iDep].dep->depositWithin(conesize);
    }
    answer.valFloat = dephlt;
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
