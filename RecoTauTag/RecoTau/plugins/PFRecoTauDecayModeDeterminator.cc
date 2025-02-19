/* class PFRecoTauDecayModeDeterminator
 *
 * Takes PFCandidates from PFTau and reconstructs tau decay mode.
 * Notably, merges photons (PFGammas) into pi zeros.
 * PFChargedHadrons are assumed to be charged pions.
 * Output candidate collections are owned (as shallow clones) by this object.
 * 
 * author: Evan K. Friis, UC Davis (evan.klose.friis@cern.ch) 
 */

#include "DataFormats/TauReco/interface/PFTauTagInfo.h"
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "RecoTauTag/TauTagTools/interface/TauTagTools.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoTauTag/TauTagTools/interface/PFCandCommonVertexFitter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "CommonTools/CandUtils/interface/AddFourMomenta.h"

#include "CLHEP/Random/RandGauss.h"

#include <memory>
#include <algorithm>

using namespace reco;
using namespace edm;
using namespace std;
typedef reco::Particle::LorentzVector LorentzVector;

class PFRecoTauDecayModeDeterminator : public EDProducer {
 public:

  typedef std::list<CompositeCandidate>                    compCandList;
  typedef std::list<CompositeCandidate>::reverse_iterator  compCandRevIter;

  void mergePiZeroes(compCandList&, compCandRevIter);
  void mergePiZeroesByBestMatch(compCandList&);

  explicit PFRecoTauDecayModeDeterminator(const edm::ParameterSet& iConfig);
  ~PFRecoTauDecayModeDeterminator();
  virtual void produce(edm::Event&,const edm::EventSetup&);

 protected:
  const double chargedPionMass;
  const double neutralPionMass;

  struct  gammaMatchContainer {
     double matchQuality;
     size_t firstIndex;
     size_t secondIndex;
  };

  static bool gammaMatchSorter (const gammaMatchContainer& first, const gammaMatchContainer& second);

 private:
  PFCandCommonVertexFitterBase* vertexFitter_;
  edm::InputTag              PFTauProducer_;
  AddFourMomenta        addP4;
  uint32_t              maxPhotonsToMerge_;             //number of photons allowed in a merged pi0
  double                maxPiZeroMass_;             
  bool                  mergeLowPtPhotonsFirst_;
  bool                  mergeByBestMatch_;
  bool                  setChargedPionMass_;
  bool                  setPi0Mass_;
  bool                  setMergedPi0Mass_;
  bool                  refitTracks_;
  bool                  filterTwoProngs_;
  bool                  filterPhotons_;  
  double                minPtFractionForSecondProng_;   //2 prongs whose second prong falls under 
  double                minPtFractionSinglePhotons_; 
  double                minPtFractionPiZeroes_; 
  TauTagTools::sortByDescendingPt<CompositeCandidate>   candDescendingSorter;
  TauTagTools::sortByAscendingPt<CompositeCandidate>    candAscendingSorter;
};

PFRecoTauDecayModeDeterminator::PFRecoTauDecayModeDeterminator(const edm::ParameterSet& iConfig):chargedPionMass(0.13957),neutralPionMass(0.13497){
  PFTauProducer_                = iConfig.getParameter<edm::InputTag>("PFTauProducer");
  maxPhotonsToMerge_            = iConfig.getParameter<uint32_t>("maxPhotonsToMerge");
  maxPiZeroMass_                = iConfig.getParameter<double>("maxPiZeroMass");             
  mergeLowPtPhotonsFirst_       = iConfig.getParameter<bool>("mergeLowPtPhotonsFirst");
  mergeByBestMatch_             = iConfig.getParameter<bool>("mergeByBestMatch");
  setChargedPionMass_           = iConfig.getParameter<bool>("setChargedPionMass");
  setPi0Mass_                   = iConfig.getParameter<bool>("setPi0Mass");
  setMergedPi0Mass_             = iConfig.getParameter<bool>("setMergedPi0Mass");
  refitTracks_                  = iConfig.getParameter<bool>("refitTracks");
  filterTwoProngs_              = iConfig.getParameter<bool>("filterTwoProngs");
  filterPhotons_                = iConfig.getParameter<bool>("filterPhotons");
  minPtFractionForSecondProng_  = iConfig.getParameter<double>("minPtFractionForSecondProng");
  minPtFractionSinglePhotons_   = iConfig.getParameter<double>("minPtFractionSinglePhotons");
  minPtFractionPiZeroes_        = iConfig.getParameter<double>("minPtFractionPiZeroes");
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
      float differenceToTruePiZeroMass          = std::abs(combinationCandidateMass - neutralPionMass);
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

bool 
PFRecoTauDecayModeDeterminator::gammaMatchSorter(const gammaMatchContainer& first,
                                                 const gammaMatchContainer& second)
{
   return (first.matchQuality < second.matchQuality);
}

void PFRecoTauDecayModeDeterminator::mergePiZeroesByBestMatch(compCandList& input)
{
   if(!input.size()) //nothing to merge... (NOTE: this line is necessary, as for size_t x, x in [0, +inf), x < -1 = true)
      return;

   std::vector<compCandList::iterator> gammas;       // iterators to all the gammas.  needed as we are using a list for compatability
                                                // with the original merging algorithm, and this implementation requires random access
   std::vector<gammaMatchContainer> matches;

   // populate the list of gammas
   for(compCandList::iterator iGamma = input.begin(); iGamma != input.end(); ++iGamma)
      gammas.push_back(iGamma);


   for(size_t gammaA = 0; gammaA < gammas.size()-1; ++gammaA)
   {
      for(size_t gammaB = gammaA+1; gammaB < gammas.size(); ++gammaB)
      {
         //construct invariant mass of this pair
         LorentzVector piZeroAB = gammas[gammaA]->p4() + gammas[gammaB]->p4();
         //different to true pizero mass
         double piZeroABMass               = piZeroAB.M();
         double differenceToTruePiZeroMass = std::abs(piZeroABMass - neutralPionMass);

         if(piZeroABMass < maxPiZeroMass_)
         {
            gammaMatchContainer   aMatch;
            aMatch.matchQuality = differenceToTruePiZeroMass;
            aMatch.firstIndex   = gammaA;
            aMatch.secondIndex  = gammaB;
            matches.push_back(aMatch);
         }
      }
   }

   sort(matches.begin(), matches.end(), gammaMatchSorter);
   //the pairs whose mass is closest to the true pi0 mass are now at the beginning
   //of this vector

   for(std::vector<gammaMatchContainer>::iterator iMatch  = matches.begin(); 
                                             iMatch != matches.end();
                                           ++iMatch)
   {
      size_t gammaA = iMatch->firstIndex;
      size_t gammaB = iMatch->secondIndex;
      //check to see that both gammas in this match have not been used (ie their iterators set to input.end())
      if( gammas[gammaA] != input.end() && gammas[gammaB] != input.end() )
      {
         //merge the second gamma into the first; loop occurs in case of multiple gamma merging option
         for(size_t bDaughter = 0; bDaughter < gammas[gammaB]->numberOfDaughters(); ++bDaughter)
            gammas[gammaA]->addDaughter( *(gammas[gammaB]->daughter(bDaughter)) );
         //update the four vector information
         addP4.set(*gammas[gammaA]);
         //delete gammaB from the list of photons/pi zeroes, as it has been merged into gammaA
         input.erase(gammas[gammaB]);
         //mark both as "merged"
         gammas[gammaA] = input.end();
         gammas[gammaB] = input.end();
      } // else this match contains a photon that has already been merged
   }

}

void PFRecoTauDecayModeDeterminator::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){

  edm::ESHandle<TransientTrackBuilder> myTransientTrackBuilder;
  edm::ESHandle<MagneticField> myMF;

  if (refitTracks_)
  {
     iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",myTransientTrackBuilder);
     iSetup.get<IdealMagneticFieldRecord>().get(myMF);
  }

  edm::Handle<PFTauCollection> thePFTauCollection;
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
     std::vector<ShallowCloneCandidate>    chargedCandidates;
     std::list<CompositeCandidate>         gammaCandidates;
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
           if(setChargedPionMass_)
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
           if(setChargedPionMass_)
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

     if (mergeByBestMatch_)
        mergePiZeroesByBestMatch(gammaCandidates);
     else
        mergePiZeroes(gammaCandidates, gammaCandidates.rbegin());

     if (filterPhotons_)
     {
        //sort by pt, from high to low.
        gammaCandidates.sort(candAscendingSorter);

        compCandRevIter wimp = gammaCandidates.rbegin();

        bool doneFiltering = false;
        while(!doneFiltering && wimp != gammaCandidates.rend())
        {
           double ptFraction          = wimp->pt()/totalFourVector.pt();
           size_t numberOfPhotons     = wimp->numberOfDaughters();

           //check if it is a single photon or has been merged
           if ( (numberOfPhotons == 1 && ptFraction < minPtFractionSinglePhotons_) ||
                (numberOfPhotons  > 1 && ptFraction < minPtFractionPiZeroes_     )    )
           {
              //remove
              totalFourVector -= wimp->p4();
              for(size_t iDaughter = 0; iDaughter < numberOfPhotons; ++iDaughter)
              {
                 filteredStuff.addDaughter(ShallowCloneCandidate(CandidateBaseRef( wimp->daughter(iDaughter)->masterClone() )));
              }

              //move to the next photon to filter
              ++wimp;
           } else
           {
              //if this pizero passes the filter, we are done looking
              doneFiltering = true;
           }
        }
        //delete the filtered objects
        gammaCandidates.erase(wimp.base(), gammaCandidates.end());
     }


     CompositeCandidate mergedPiZerosToAdd;
     for( std::list<CompositeCandidate>::iterator iGamma  = gammaCandidates.begin();
                                             iGamma != gammaCandidates.end();
                                           ++iGamma)
     {
        if (setPi0Mass_) // set mass as pi 0
        {
           if (iGamma->numberOfDaughters() == 1) // for merged gamma pairs, check if user wants to keep ECAL mass
              iGamma->setMass(neutralPionMass);
           else if (setMergedPi0Mass_)
              iGamma->setMass(neutralPionMass);
        }
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
DEFINE_FWK_MODULE(PFRecoTauDecayModeDeterminator);
