#include "RecoJets/JetAlgorithms/interface/CMSInsideOutAlgorithm.h"

#include "RecoJets/JetAlgorithms/interface/CompoundPseudoJet.h"


using namespace std;

void
CMSInsideOutAlgorithm::run(const std::vector<fastjet::PseudoJet>& fInput, std::vector<fastjet::PseudoJet> & fOutput)
{

   //make a list of input objects
   list<fastjet::PseudoJet> input;
   for (std::vector<fastjet::PseudoJet>::const_iterator candIter  = fInput.begin();
	candIter != fInput.end(); 
	++candIter) {
      input.push_back(*candIter);
   }

   while( !input.empty() && input.front().perp() > seedThresholdPt_ ) 
   {
      //get seed eta/phi
      double seedEta = input.front().eta();
      double seedPhi = input.front().phi();

      //find iterators to those objects that are in the max cone size
      list<inputListIter> maxCone;
      // add seed, then test elements after seed
      inputListIter iCand = input.begin();
      maxCone.push_back(iCand++);
      for(; iCand != input.end(); ++iCand)
      {
	const fastjet::PseudoJet& candidate = *iCand;
         if( reco::deltaR2(seedEta, candidate.eta(), seedPhi, candidate.phi()) < maxSizeSquared_ )
            maxCone.push_back(iCand);
      }
      //sort objects by increasing DR about the seed  directions
      maxCone.sort(ListIteratorLesserByDeltaR(seedEta, seedPhi));
      list<inputListIter>::const_iterator position = maxCone.begin(); 
      bool limitReached = false;
      double totalET    = (**position).perp();
      ++position;
      while(position != maxCone.end() && !limitReached)
      {
 	 const fastjet::PseudoJet& theCandidate = **position;
         double candidateET    = theCandidate.perp() + totalET; 
         double candDR2        = reco::deltaR2(seedEta, theCandidate.eta(), seedPhi, theCandidate.phi());
         if( candDR2 < minSizeSquared_ ||  candDR2*candidateET*candidateET < growthParameterSquared_ )
            totalET = candidateET;
         else
            limitReached = true;
         ++position;
      }
      //turn this into a final jet
      fastjet::PseudoJet final;
      for(list<inputListIter>::const_iterator iNewJet  = maxCone.begin();
                                              iNewJet != position;
                                            ++iNewJet)
      {
   	 final += **iNewJet;
         input.erase(*iNewJet);
      }
      fOutput.push_back(final);
   } // end loop over seeds
   GreaterByEtPseudoJet compJets;
   sort (fOutput.begin (), fOutput.end (), compJets);
}
         

