#include "HiggsAnalysis/CombinedLimit/interface/TestProposal.h"
#include <RooArgSet.h>
#include <iostream>
#include <memory>
#include <TIterator.h>
#include <RooStats/RooStatsUtils.h>


// Populate xPrime with a new proposed point
void TestProposal::Propose(RooArgSet& xPrime, RooArgSet& x )
{
   static int tries = 0;
   RooStats::RandomizeCollection(xPrime);
   if (tries++ < 20) {  
        std::cout << "TestProposal::Propose: x'" << std::endl; xPrime.Print("V"); 
        std::cout << "TestProposal::Propose: x " << std::endl; x.Print("V"); 
   }
}

// Return the probability of proposing the point x1 given the starting
// point x2
Double_t TestProposal::GetProposalDensity(RooArgSet& x1,
                                          RooArgSet& x2)
{
   static int tries = 0;
   if (tries++ < 20) {  
       std::cout << "TestProposal::GetProposalDensity: x1" << std::endl; x1.Print("V"); 
       std::cout << "TestProposal::GetProposalDensity: x2" << std::endl; x2.Print("V"); 
   }
   // For a uniform proposal, all points have equal probability and the
   // value of the proposal density function is:
   // 1 / (N-dimensional volume of interval)
   Double_t volume = 1.0;
   std::auto_ptr<TIterator> it(x1.createIterator());
   RooRealVar* var;
   while ((var = (RooRealVar*)it->Next()) != NULL)
      volume *= (var->getMax() - var->getMin());
   return 1.0 / volume;
}

ClassImp(TestProposal)
