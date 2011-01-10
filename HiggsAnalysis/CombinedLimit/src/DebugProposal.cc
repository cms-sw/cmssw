#include "HiggsAnalysis/CombinedLimit/interface/DebugProposal.h"
#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <iostream>
#include <memory>
#include <RooStats/RooStatsUtils.h>

DebugProposal::DebugProposal(RooStats::ProposalFunction *p, RooAbsPdf *pdf, RooAbsData *data, int tries) : 
    RooStats::ProposalFunction(), prop_(p), pdf_(pdf), tries_(tries) 
{
    if (pdf && data) {
        nll_.reset(pdf->createNLL(*data));
        RooArgSet *par = pdf->getParameters(*data);
        RooStats::RemoveConstantParameters(par);
        params_.add(*par);
        delete par;
    }
    if (tries) {
        p->Print("V");
    }
}

// Populate xPrime with a new proposed point
void DebugProposal::Propose(RooArgSet& xPrime, RooArgSet& x )
{
   prop_->Propose(xPrime,x);
   if (tries_ > 0) {  
        std::cout << "DebugProposal::Propose: x'" << std::endl; xPrime.Print("V"); 
        std::cout << "DebugProposal::Propose: x " << std::endl; x.Print("V"); 
        if (nll_.get()) {
            RooStats::SetParameters(&xPrime, &params_);
            double nllXP = nll_->getVal();
            RooStats::SetParameters(&x, &params_);
            double nllX  = nll_->getVal();
            std::cout << "DebugProposal::Propose: NLL(x)  = " << nllX  << std::endl;
            std::cout << "DebugProposal::Propose: NLL(x') = " << nllXP << std::endl;
            std::cout << "DebugProposal::Propose: uncorrected acc prob. = " << exp(nllX - nllXP) << std::endl;
            std::cout << "DebugProposal::Propose: tracing NLL(x')" << std::endl;
            RooStats::SetParameters(&xPrime, &params_);
            tracePdf(nll_.get());
            std::cout << "DebugProposal::Propose: tracing NLL(x)" << std::endl;
            RooStats::SetParameters(&x, &params_);
            tracePdf(nll_.get());
        }
        --tries_;
   }
}

// Return the probability of proposing the point x1 given the starting
// point x2
Double_t DebugProposal::GetProposalDensity(RooArgSet& x1,
                                          RooArgSet& x2)
{
   if (tries_ > 0) {  
       std::cout << "DebugProposal::GetProposalDensity: x1" << std::endl; x1.Print("V"); 
       std::cout << "DebugProposal::GetProposalDensity: x2" << std::endl; x2.Print("V"); 
   }
   Double_t ret = prop_->GetProposalDensity(x1,x2);
   if (tries_ > 0) { std::cout << "DebugProposal::GetProposalDensity: return " << ret << std::endl; }
   return ret;
}


Bool_t DebugProposal::IsSymmetric(RooArgSet& x1, RooArgSet& x2) {
   if (tries_ > 0) {  
       std::cout << "DebugProposal::IsSymmetric: x1" << std::endl; x1.Print("V"); 
       std::cout << "DebugProposal::IsSymmetric: x2" << std::endl; x2.Print("V"); 
   }
   Bool_t ret = prop_->IsSymmetric(x1,x2);
   if (tries_ > 0) { std::cout << "DebugProposal::IsSymmetric: return " << ret << std::endl; }
   return ret;
}

void DebugProposal::tracePdf(RooAbsReal *pdf) {
   RooArgList deps;
   deps.add(*pdf);
   pdf->treeNodeServerList(&deps);
   for (int i = 0, n = deps.getSize(); i < n; ++i) {
      RooAbsReal *rar = dynamic_cast<RooAbsReal *>(deps.at(i));
      if (typeid(*rar) == typeid(RooRealVar) || rar->isConstant()) continue;
      if (rar != 0) {
        std::cout << "   " << rar->GetName() << " = " << rar->getVal() << std::endl;
      }
   }
}

ClassImp(DebugProposal)
