#include "../interface/TestProposal.h"
#include <RooArgSet.h>
#include <iostream>
#include <memory>
#include <TIterator.h>
#include <RooRandom.h>
#include <RooStats/RooStatsUtils.h>

TestProposal::TestProposal(double divisor, const RooRealVar *alwaysStepMe) : 
    RooStats::ProposalFunction(),
    divisor_(1./divisor),
    poiDivisor_(divisor_)
{
    alwaysStepMe_.add(*alwaysStepMe);
}
     
TestProposal::TestProposal(double divisor, const RooArgList &alwaysStepMe) : 
    RooStats::ProposalFunction(),
    divisor_(1./divisor),
    poiDivisor_(divisor_),
    alwaysStepMe_(alwaysStepMe)
{
    if (alwaysStepMe.getSize() > 1) poiDivisor_ /= sqrt(double(alwaysStepMe.getSize()));
}
 

// Populate xPrime with a new proposed point
void TestProposal::Propose(RooArgSet& xPrime, RooArgSet& x )
{
   RooStats::SetParameters(&x, &xPrime);
   RooLinkedListIter it(xPrime.iterator());
   RooRealVar* var;
   int n = xPrime.getSize(), j = floor(RooRandom::uniform()*n);
   for (int i = 0; (var = (RooRealVar*)it.Next()) != NULL; ++i) {
      if (i == j) {
        if (alwaysStepMe_.contains(*var)) break; // don't step twice
        double val = var->getVal(), max = var->getMax(), min = var->getMin(), len = max - min;
        val += RooRandom::gaussian() * len * divisor_;
        while (val > max) val -= len;
        while (val < min) val += len;
        var->setVal(val);
        break;
      }
   }
   it = alwaysStepMe_.iterator();
   for (RooRealVar *poi = (RooRealVar*)it.Next(); poi != NULL; poi = (RooRealVar*)it.Next()) {
        RooRealVar *var = (RooRealVar*) xPrime.find(poi->GetName());
        double val = var->getVal(), max = var->getMax(), min = var->getMin(), len = max - min;
        val += RooRandom::gaussian() * len * poiDivisor_;
        while (val > max) val -= len;
        while (val < min) val += len;
        var->setVal(val);
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
