#include "RecoTauTag/RecoTau/interface/CMSInsideOutAlgorithm.h"

void
CMSInsideOutAlgorithm::run(const JetReco::InputCollection& fInput, JetReco::OutputCollection* fOutput) const
{
   if(!fOutput) return;
   //make a list of input objects ordered by ET
   list<InputItem> input;
   for (InputCollection::const_iterator candIter  = fInput.begin();
                                        candIter != fInput.end(); 
                                      ++candIter) 
      input.push_back(*candIter);

   GreaterByEtRef <InputItem> compCandidate;
   input.sort(compCandidate);
   while( !input.empty() && input.front()->et() > seedThresholdPt_ ) 
   {
      //get seed eta/phi
      double seedEta = input.front()->eta();
      double seedPhi = input.front()->phi();

      //find iterators to those objects that are in the max cone size
      list<inputListIter> maxCone;
      // add seed, then test elements after seed
      inputListIter iCand = input.begin();
      maxCone.push_back(iCand++);
      for(; iCand != input.end(); ++iCand)
      {
         const InputItem& candidate = *iCand;
         if( reco::deltaR2(seedEta, candidate->eta(), seedPhi, candidate->phi()) < maxSizeSquared_ )
            maxCone.push_back(iCand);
      }
      //sort objects by increasing DR about the seed  directions
      maxCone.sort(ListIteratorLesserByDeltaR(seedEta, seedPhi));
      list<inputListIter>::const_iterator position = maxCone.begin(); 
      bool limitReached = false;
      double totalET    = (**position)->et();
      ++position;
      while(position != maxCone.end() && !limitReached)
      {
         const InputItem& theCandidate = **position;
         double candidateET    = theCandidate->et() + totalET;
         double candDR2        = reco::deltaR2(seedEta, theCandidate->eta(), seedPhi, theCandidate->phi());
         if( candDR2 < minSizeSquared_ ||  candDR2*candidateET*candidateET < growthParameterSquared_ )
            totalET = candidateET;
         else
            limitReached = true;
         ++position;
      }
      //turn this into a final jet
      InputCollection jetConstituents;
      for(list<inputListIter>::const_iterator iNewJet  = maxCone.begin();
                                              iNewJet != position;
                                            ++iNewJet)
      {
         jetConstituents.push_back(**iNewJet);
         input.erase(*iNewJet);
      }
      fOutput->push_back( ProtoJet(jetConstituents) );
   } // end loop over seeds
   GreaterByPt<ProtoJet> compJets;
   sort (fOutput->begin (), fOutput->end (), compJets);
}
         

