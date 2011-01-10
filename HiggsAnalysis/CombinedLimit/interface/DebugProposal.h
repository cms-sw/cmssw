#ifndef HiggsAnalysis_CombinedLimit_DebugProposal_h
#define HiggsAnalysis_CombinedLimit_DebugProposal_h

#include <memory>
#include <Rtypes.h>

class RooAbsData;
#include <RooArgSet.h>
#include <RooAbsReal.h>
#include <RooAbsPdf.h>

#include <RooStats/ProposalFunction.h>

class DebugProposal : public RooStats::ProposalFunction {

   public:
      DebugProposal() : RooStats::ProposalFunction(), prop_(0), nll_(0), params_(), tries_(0)  {}
      DebugProposal(RooStats::ProposalFunction *p, RooAbsPdf *pdf, RooAbsData *data, int tries) ;

      // Populate xPrime with a new proposed point
      virtual void Propose(RooArgSet& xPrime, RooArgSet& x);

      // Determine whether or not the proposal density is symmetric for
      // points x1 and x2 - that is, whether the probabilty of reaching x2
      // from x1 is equal to the probability of reaching x1 from x2
      virtual Bool_t IsSymmetric(RooArgSet& x1, RooArgSet& x2) ;

      // Return the probability of proposing the point x1 given the starting
      // point x2
      virtual Double_t GetProposalDensity(RooArgSet& x1, RooArgSet& x2);

      virtual ~DebugProposal() {}

      ClassDef(DebugProposal,1) // A concrete implementation of ProposalFunction, that uniformly samples the parameter space.

    private:
        RooStats::ProposalFunction *prop_;
        std::auto_ptr<RooAbsReal> nll_;
        RooAbsPdf                 *pdf_;
        RooArgSet                 params_;
        int  tries_;
        void tracePdf(RooAbsReal *pdf) ;
};

#endif
