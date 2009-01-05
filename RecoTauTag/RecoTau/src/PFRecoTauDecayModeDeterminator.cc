#include "RecoTauTag/RecoTau/interface/PFRecoTauDecayModeDeterminator.h"

PFRecoTauDecayModeDeterminator::PFRecoTauDecayModeDeterminator(const ParameterSet& iConfig):chargedPionMass(0.13957),neutralPionMass(0.13497){
  PFTauProducer_                = iConfig.getParameter<InputTag>("PFTauProducer");
  maxPhotonsToMerge_            = iConfig.getParameter<uint32_t>("maxPhotonsToMerge");
  maxPiZeroMass_                = iConfig.getParameter<double>("maxPiZeroMass");             
  mergeLowPtPhotonsFirst_       = iConfig.getParameter<bool>("mergeLowPtPhotonsFirst");
  refitTracks_                  = iConfig.getParameter<bool>("refitTracks");
  filterTwoProngs_              = iConfig.getParameter<bool>("filterTwoProngs");
  filterPhotons_                = iConfig.getParameter<bool>("filterPhotons");
  minPtFractionForSecondProng_  = iConfig.getParameter<double>("minPtFractionForSecondProng");
  minPtFractionForGammas_       = iConfig.getParameter<double>("minPtFractionForThirdGamma");
  //setup vertex fitter
  vertexFitter_ = new PFCandCommonVertexFitter<KalmanVertexFitter>(iConfig);
  produces<PFTauDecayModeAssociation>();      
}

PFRecoTauDecayModeDeterminator::~PFRecoTauDecayModeDeterminator()
{
//   delete vertexFitter_;  //now a very small memory leak, fix me later
}


/* 
 * ******************************************************************
   **     Merges a list of photons in to Pi0 candidates            **
   ******************************************************************
 */
void PFRecoTauDecayModeDeterminator::mergePiZeroes(compCandList& input, compCandRevIter seed)
{
   //uses std::list instead of vector, so that iterators can be deleted in situ
   //we go backwards for historical reasons ;)
   if(seed == input.rend())
      return;
   compCandRevIter bestMatchSoFar;
   LorentzVector combinationCandidate;
   float closestInvariantMassDifference = maxPiZeroMass_ + 1;
   bool foundACompatibleMatch = false;
   //find the best match to make a pi0
   compCandRevIter potentialMatch = seed;
   ++potentialMatch;
   for(; potentialMatch != input.rend(); ++potentialMatch)
   {
      // see how close this combination comes to the pion mass
      LorentzVector seedFourVector              = seed->p4();
      LorentzVector toAddFourVect               = potentialMatch->p4();
      combinationCandidate                      = seedFourVector + toAddFourVect;
      float combinationCandidateMass            = combinationCandidate.M();
      float differenceToTruePiZeroMass          = abs(combinationCandidateMass - neutralPionMass);
      if(combinationCandidateMass < maxPiZeroMass_ && differenceToTruePiZeroMass < closestInvariantMassDifference)
      {
         closestInvariantMassDifference = differenceToTruePiZeroMass;
         bestMatchSoFar = potentialMatch;
         foundACompatibleMatch = true;
      }
   }
   //if we found a combination that might make a pi0, combine it into the seed gamma, erase it, then see if we can add anymore
   if(foundACompatibleMatch && seed->numberOfDaughters() < maxPhotonsToMerge_)
   {
      //combine match into Seed and update four vector
      if(bestMatchSoFar->numberOfDaughters() > 0)
      {
         const Candidate* photonToAdd = (*bestMatchSoFar).daughter(0);
         seed->addDaughter(*photonToAdd);
      }
      addP4.set(*seed);
      //remove match as it is now contained in the seed 
      input.erase( (++bestMatchSoFar).base() );  //convert to normal iterator, after correct for offset
      mergePiZeroes(input, seed);
   } else
   {
      // otherwise move to next highest object and recurse
      addP4.set(*seed);
      ++seed;
      mergePiZeroes(input, seed);
   }
}

void PFRecoTauDecayModeDeterminator::produce(Event& iEvent,const EventSetup& iSetup){

  ESHandle<TransientTrackBuilder> myTransientTrackBuilder;
  ESHandle<MagneticField> myMF;

  if (refitTracks_)
  {
     iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",myTransientTrackBuilder);
     iSetup.get<IdealMagneticFieldRecord>().get(myMF);
  }

  Handle<PFTauCollection> thePFTauCollection;
  iEvent.getByLabel(PFTauProducer_,thePFTauCollection);

  auto_ptr<PFTauDecayModeAssociation> result(new PFTauDecayModeAssociation(PFTauRefProd(thePFTauCollection)));

  size_t numberOfPFTaus = thePFTauCollection->size();
  for(size_t iPFTau = 0; iPFTau < numberOfPFTaus; ++iPFTau)
  {
     //get the reference to the PFTau
     PFTauRef           pfTauRef(thePFTauCollection, iPFTau);
     PFTau              myPFTau = *pfTauRef;
     
     //get the charged & neutral collections corresponding to this PFTau
     const PFCandidateRefVector& theChargedHadronCandidates = myPFTau.signalPFChargedHadrCands();
     const PFCandidateRefVector& theGammaCandidates         = myPFTau.signalPFGammaCands();

     LorentzVector totalFourVector;                       //contains non-filtered stuff only.

     //shallow clone everything
     vector<ShallowCloneCandidate>    chargedCandidates;
     list<CompositeCandidate>         gammaCandidates;
     VertexCompositeCandidate         chargedCandsToAdd;  
     CompositeCandidate               filteredStuff;      //empty for now.

     bool needToProcessTracks = true;
     if (filterTwoProngs_ && theChargedHadronCandidates.size() == 2)
     {
        size_t indexOfHighestPt = (theChargedHadronCandidates[0]->pt() > theChargedHadronCandidates[1]->pt()) ? 0 : 1;
        size_t indexOfLowerPt   = ( indexOfHighestPt ) ? 0 : 1; 
        //maybe include a like signed requirement?? (future)
        double highPt = theChargedHadronCandidates[indexOfHighestPt]->pt();
        double lowPt  = theChargedHadronCandidates[indexOfLowerPt]->pt();
        if (lowPt/highPt < minPtFractionForSecondProng_)  //if it is super low, filter it!
        {
           needToProcessTracks = false;  //we are doing it here instead
           chargedCandsToAdd.addDaughter(ShallowCloneCandidate(CandidateBaseRef(theChargedHadronCandidates[indexOfHighestPt])));
           Candidate* justAdded = chargedCandsToAdd.daughter(chargedCandsToAdd.numberOfDaughters()-1);
           totalFourVector += justAdded->p4();
           justAdded->setMass(chargedPionMass);
           //add the two prong to the list of filtered stuff (to be added to the isolation collection later)
           filteredStuff.addDaughter(ShallowCloneCandidate(CandidateBaseRef(theChargedHadronCandidates[indexOfLowerPt])));
        }
     }

     if(needToProcessTracks) //not a two prong, filter is turned off, or 2nd prong passes cuts
     {
        for( PFCandidateRefVector::const_iterator iCharged  = theChargedHadronCandidates.begin();
              iCharged != theChargedHadronCandidates.end();
              ++iCharged)
        {
           // copy as shallow clone, and asssume mass of pi+
           chargedCandsToAdd.addDaughter(ShallowCloneCandidate(CandidateBaseRef(*iCharged)));
           Candidate* justAdded = chargedCandsToAdd.daughter(chargedCandsToAdd.numberOfDaughters()-1);
           totalFourVector += justAdded->p4();
           justAdded->setMass(chargedPionMass);
        }
     }

     for( PFCandidateRefVector::const_iterator iGamma  = theGammaCandidates.begin();
                                               iGamma != theGammaCandidates.end();
                                             ++iGamma)
     {
        CompositeCandidate potentialPiZero;
        potentialPiZero.addDaughter(ShallowCloneCandidate(CandidateBaseRef(*iGamma)));
        addP4.set(potentialPiZero);
        totalFourVector += potentialPiZero.p4();
        gammaCandidates.push_back(potentialPiZero);
     }

     //sort the photons by pt before passing to merger
     if (mergeLowPtPhotonsFirst_)
        gammaCandidates.sort(candDescendingSorter);
     else
        gammaCandidates.sort(candAscendingSorter);

     mergePiZeroes(gammaCandidates, gammaCandidates.rbegin());

     CompositeCandidate mergedPiZerosToAdd;
     for( list<CompositeCandidate>::iterator iGamma  = gammaCandidates.begin();
                                             iGamma != gammaCandidates.end();
                                           ++iGamma)
     {
        iGamma->setMass(neutralPionMass);
        mergedPiZerosToAdd.addDaughter(*iGamma);
     }

     // apply vertex fitting.
     if (refitTracks_ && chargedCandsToAdd.numberOfDaughters() > 1)
     {
        vertexFitter_->set(myMF.product());
        vertexFitter_->set(chargedCandsToAdd);  //refits tracks, adds vertexing info
     }

     // correctly set the four vectors of the composite candidates
     addP4.set(chargedCandsToAdd);
     addP4.set(mergedPiZerosToAdd);
     addP4.set(filteredStuff);

     /*
     LorentzVector refitFourVector = chargedCandsToAdd.p4() + mergedPiZerosToAdd.p4();

     edm::LogInfo("PFTauDecayModeDeterminator") << "Found nCharged: " << chargedCandsToAdd.numberOfDaughters()
                                  << " and nNeutral: " << mergedPiZerosToAdd.numberOfDaughters()
                                  << " Former mass: " << totalFourVector.mass() 
                                  << " New mass: " << refitFourVector.mass();
     */

     PFTauDecayMode myDecayModeTau(chargedCandsToAdd, mergedPiZerosToAdd, filteredStuff);
     myDecayModeTau.setPFTauRef(pfTauRef);
     result->setValue(iPFTau, myDecayModeTau);
  }
  iEvent.put(result);
}
