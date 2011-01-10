#include "HiggsAnalysis/CombinedLimit/interface/TestProposal.h"
#include <RooArgSet.h>
#include <iostream>
#include <memory>
#include <TIterator.h>
#include <RooRandom.h>
#include <RooStats/RooStatsUtils.h>

TestProposal::TestProposal(double divisor) : 
    RooStats::ProposalFunction(),
    divisor_(1./divisor)
{
}
     

// Populate xPrime with a new proposed point
void TestProposal::Propose(RooArgSet& xPrime, RooArgSet& x )
{
   RooStats::SetParameters(&x, &xPrime);
   std::auto_ptr<TIterator> it(xPrime.createIterator());
   RooRealVar* var;
   int n = xPrime.getSize(), j = floor(RooRandom::uniform()*n);
   for (int i = 0; (var = (RooRealVar*)it->Next()) != NULL; ++i) {
      if (i == j) {
        double val = var->getVal(), max = var->getMax(), min = var->getMin(), len = max - min;
        val += RooRandom::gaussian() * len * divisor_;
        while (val > max) val -= len;
        while (val < min) val += len;
        var->setVal(val);
        //std::cout << "Proposing a step along " << var->GetName() << std::endl;
      }
   }
}

Bool_t TestProposal::IsSymmetric(RooArgSet& x1, RooArgSet& x2) {
   return true;
}

// Return the probability of proposing the point x1 given the starting
// point x2
Double_t TestProposal::GetProposalDensity(RooArgSet& x1,
                                          RooArgSet& x2)
{
   return 1.0; // should not be needed
}

ClassImp(TestProposal)
