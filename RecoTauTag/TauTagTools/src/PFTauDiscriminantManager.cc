#include "RecoTauTag/TauTagTools/interface/PFTauDiscriminantManager.h"

namespace PFTauDiscriminants
{
using namespace std;
using namespace reco;

typedef std::vector<const reco::Candidate*> candPtrVector;

PFTauDiscriminantManager::PFTauDiscriminantManager()
{
   iAmSignal_           = false;
   iAmNull_             = false;
   eventWeight_         = 1.0;
   currentTauDecayMode_ = NULL;
   eventData_           = NULL;
   mainTrack_           = NULL;
}

void
PFTauDiscriminantManager::addDiscriminant(Discriminant* const discriminant)
{
   if (!discriminant)
   {
      edm::LogError("PFTauDiscriminantManager") << "Error adding a discriminant, null pointer!";
      return;
   }
   string            discriminantName = discriminant->name();
   myDiscriminants_.insert(make_pair(discriminantName, discriminant));
}

void 
PFTauDiscriminantManager::clearCache()
{
   mainTrack_ = NULL;
   signalObjectsSortedByPt_.clear();
   signalObjectsSortedByDR_.clear();
   outlierObjectsSortedByPt_.clear();
   outlierObjectsSortedByDR_.clear();
}

bool
PFTauDiscriminantManager::setTau(const reco::PFTauDecayMode& theTau, bool prePass, bool preFail)
{
   currentTauDecayMode_ = &theTau;
   iAmNull_             = false;
   prePass_             = prePass;
   preFail_             = preFail;
   //reset cached collections
   clearCache();
   
   for(discriminantHolder::iterator aDiscriminant  = myDiscriminants_.begin();
                                    aDiscriminant != myDiscriminants_.end();
                                  ++aDiscriminant)
   {
      Discriminant* const theAlgo  = aDiscriminant->second;
      if (!theAlgo)
      {
         string theName  = aDiscriminant->first;
         edm::LogError("PFTauDiscriminantManager") << "Error filling discriminant " << theName <<", null pointer!";
         return false;
      }
      theAlgo->compute(this);
   }
   return true;
}

void 
PFTauDiscriminantManager::setEvent(const edm::Event& iEvent, double eventWeight)
{
   eventData_           = &iEvent;
   eventWeight_         = eventWeight;
}

bool
PFTauDiscriminantManager::setNullResult()
{
   currentTauDecayMode_ = NULL;
   iAmNull_             = true;
   prePass_             = false;
   preFail_             = false;
   //reset cached collections
   clearCache();

   for(discriminantHolder::iterator aDiscriminant  = myDiscriminants_.begin();
                                    aDiscriminant != myDiscriminants_.end();
                                  ++aDiscriminant)
   {
      Discriminant* const theAlgo  = aDiscriminant->second;
      if (!theAlgo)
      {
         string theName  = aDiscriminant->first;
         edm::LogError("PFTauDiscriminantManager") << "Error filling discriminant " << theName <<", null pointer!";
         return false;
      }
      theAlgo->setNullResult(this);
   }
   return true;
}



void PFTauDiscriminantManager::fillSignalObjects(candPtrVector& toFill)
{
   toFill.clear();
   if (currentTauDecayMode_ == NULL)
   {
      edm::LogError("PFTauDiscriminantManager") << "Trying to get signal objects from null PFTauDecayMode object!  Returning empty vector...";
      return;
   }
   candPtrVector tempChargedVector = currentTauDecayMode_->chargedPionCandidates();
   candPtrVector tempNeutralVector = currentTauDecayMode_->neutralPionCandidates();
   toFill.insert(toFill.end(), tempChargedVector.begin(), tempChargedVector.end());
   toFill.insert(toFill.end(), tempNeutralVector.begin(), tempNeutralVector.end());
}

void PFTauDiscriminantManager::fillOutlierObjects(candPtrVector& toFill)
{
   toFill.clear();
   if (currentTauDecayMode_ == NULL)
   {
      edm::LogError("PFTauDiscriminantManager") << "Trying to get QCD objects from null PFTauDecayMode object!  Returning empty vector...";
      return;
   }

   // add in filtered objects (created in PFRecoTauDecayModeDeterminator) i.e filtered 2-prongs
   // note that this uses the underlying PFCandidates, to be consistent w/ the rest of the objects
   PFCandidateRefVector theFilteredObjects = currentTauDecayMode_->filteredPFCandidates();
   
   for(PFCandidateRefVector::const_iterator iFilteredCand  = theFilteredObjects.begin();
                                            iFilteredCand != theFilteredObjects.end();
                                          ++iFilteredCand)
   {
      const PFCandidate* pfCand = iFilteredCand->get();
      const Candidate* castedCand = static_cast<const Candidate*>(pfCand);
      if (castedCand)
         toFill.push_back(castedCand);
   }

   // get associated PFTau from PFTauDecayMode
   const PFTau* originalTau = currentTauDecayMode_->pfTauRef().get();
   if(originalTau) //this may be null by design if there is no associated PFTau (e.g. if DecayMode is constructed from MC truth)
   {
      const PFCandidateRefVector& theOutliers = originalTau->isolationPFCands();
      for(PFCandidateRefVector::const_iterator iIsoCand  = theOutliers.begin();
                                               iIsoCand != theOutliers.end();
                                             ++iIsoCand)
      {
         const PFCandidate* pfCand = iIsoCand->get();
         const Candidate* castedCand = static_cast<const Candidate*>(pfCand);
         if (castedCand)
            toFill.push_back(castedCand);
      }
   }
}

const reco::Candidate*
PFTauDiscriminantManager::mainTrack() 
{
   if (mainTrack_ == NULL) //otherwise already cached or d.n.e
   {
      if (!this->getDecayMode())
      {
         edm::LogError("PFTauDiscriminantManager") << "In ::mainTrack(), trying to access a null PFTauDecayMode - returning null pointer for main track";
         return NULL;
      }

      std::vector<const reco::Candidate*> myChargedCandidates = getDecayMode()->chargedPionCandidates();
      size_t nTracks = myChargedCandidates.size();
      if (!nTracks) 
      {
         // ...removing this warning for now, not sure what to do about this case (as it shoudl be permissible to pass a jet->pftau->pfTauDecayMode of all gammas??)
         //edm::LogError("PFTauDiscriminantManager") << "In ::mainTrack(), associated PFTauDecayMode has no associated tracks, returning null pointer.";
         return NULL;
      }

      //if there are more than three tracks, only take the top three, by Pt
      TauTagTools::sortByAscendingPt<reco::Candidate> ptSorter;
      sort(myChargedCandidates.begin(), myChargedCandidates.end(), ptSorter);
      size_t maxTracks = (nTracks > 3) ? 3 : nTracks;
      int    charge    = 0;

      if (maxTracks < 3) //two or one track, returning higher Pt track
         mainTrack_ = myChargedCandidates[0];
      else
      {
         for(size_t iTrack = 0; iTrack < maxTracks; ++iTrack)
            charge += myChargedCandidates[iTrack]->charge();

         for(size_t iTrack = 0; iTrack < maxTracks; ++iTrack)
         {
            int currentCharge = myChargedCandidates[iTrack]->charge();
            if (currentCharge != charge)
            {
               mainTrack_ = myChargedCandidates[iTrack];
               break;
            }
         }
      }
   }
   return mainTrack_;
}
   


candPtrVector
PFTauDiscriminantManager::filterByCharge(const candPtrVector& input, bool isCharged) const
{
   candPtrVector output;
   for(candPtrVector::const_iterator iCandidate  = input.begin();
                                     iCandidate != input.end();
                                   ++iCandidate)
   {
      bool chargeType = (*iCandidate)->charge();
      if( chargeType == isCharged ) 
         output.push_back(*iCandidate);
   }
   return output;
}

const std::vector<const reco::Candidate*>&
PFTauDiscriminantManager::signalObjectsSortedByPt()
{
   // return already computed vector if has already been computed or is empty (due to null tau)
   if(!signalObjectsSortedByPt_.empty() || iAmNull_)  
   {
      return signalObjectsSortedByPt_;
   }
   else
   {  
      TauTagTools::sortByAscendingPt<reco::Candidate> mySorter;
      fillSignalObjects(signalObjectsSortedByPt_);
      sort(signalObjectsSortedByPt_.begin(), signalObjectsSortedByPt_.end(), mySorter);
   }
   return signalObjectsSortedByPt_;
}

const std::vector<const reco::Candidate*>&
PFTauDiscriminantManager::signalObjectsSortedByDR()
{
   // return already computed vector if has already been computed or is empty (due to null tau)
   if(!signalObjectsSortedByDR_.empty() || iAmNull_)  
   {
      return signalObjectsSortedByDR_;
   }
   else
   {  
      if (currentTauDecayMode_ == NULL)
      {
         edm::LogError("PFTauDiscriminantManager") << "Trying to get signal objects from null PFTauDecayMode object!  Returning empty vector...";
         return signalObjectsSortedByDR_;
      }
      math::XYZVector signalAxisVector = currentTauDecayMode_->momentum();
      TauTagTools::sortByOpeningAngleAscending<reco::Candidate> mySorter(signalAxisVector, TauTagTools::computeDeltaR);
      fillSignalObjects(signalObjectsSortedByDR_);
      sort(signalObjectsSortedByDR_.begin(), signalObjectsSortedByDR_.end(), mySorter);
   }
   return signalObjectsSortedByDR_;
}

const std::vector<const reco::Candidate*>&
PFTauDiscriminantManager::outlierObjectsSortedByPt()
{
   if(!outlierObjectsSortedByPt_.empty() || iAmNull_)
   {
      return outlierObjectsSortedByPt_;
   }
   else
   {
      fillOutlierObjects(outlierObjectsSortedByPt_);
      TauTagTools::sortByAscendingPt<reco::Candidate> mySorter;
      sort(outlierObjectsSortedByPt_.begin(), outlierObjectsSortedByPt_.end(), mySorter);
   }
   return outlierObjectsSortedByPt_;
}

const std::vector<const reco::Candidate*>&
PFTauDiscriminantManager::outlierObjectsSortedByDR()
{
   if(!outlierObjectsSortedByDR_.empty() || iAmNull_)
   {
      return outlierObjectsSortedByDR_;
   }
   else
   {
      if (currentTauDecayMode_ == NULL)
      {
         edm::LogError("PFTauDiscriminantManager") << "Trying to get outlier objects from null PFTauDecayMode object!  Returning empty vector...";
         return outlierObjectsSortedByDR_;
      }
      math::XYZVector signalAxisVector = currentTauDecayMode_->momentum();
      fillOutlierObjects(outlierObjectsSortedByDR_);
      TauTagTools::sortByOpeningAngleAscending<reco::Candidate> mySorter(signalAxisVector, TauTagTools::computeDeltaR);
      sort(outlierObjectsSortedByDR_.begin(), outlierObjectsSortedByDR_.end(), mySorter);
   }
   return outlierObjectsSortedByDR_;
}


bool
PFTauDiscriminantManager::branchTree(TTree* treeToBranch, bool addTargetBranch, bool addWeightBranch)
{
   if(!treeToBranch)
   {
      edm::LogError("PFTauDiscriminantManager") << "Error: trying to branch ttree - TTree pointer is null!";
      return false;
   }

   //add magic variables _TARGET_ (for sig/bkg) and _WEIGHT_, and ISNULL for non-existence
   if (addTargetBranch)
      treeToBranch->Branch("__TARGET__", &iAmSignal_,  "__TARGET__/O");  //needs bugfix in MVA framework code..
   if (addWeightBranch)
      treeToBranch->Branch("__WEIGHT__", &eventWeight_,"__WEIGHT__/D");
   // note: Target and Weight are normally added after the fact, in the training code.

   treeToBranch->Branch("__ISNULL__",  &iAmNull_,"__ISNULL__/O");
   treeToBranch->Branch("__PREPASS__", &prePass_,"__PREPASS__/O");
   treeToBranch->Branch("__PREFAIL__", &preFail_,"__PREFAIL__/O");

   //loop over all the variables and make a branch for each one
   for(discriminantHolder::iterator iVariable  = myDiscriminants_.begin();
                                    iVariable != myDiscriminants_.end();
                                  ++iVariable)
   {
      Discriminant * theDiscriminant = iVariable->second;
      edm::LogInfo("PFTauDiscriminantManager") << "Branching for discriminant w/ name: " << theDiscriminant->name();
      theDiscriminant->branchTree(treeToBranch);
   }
   return true;
}

void
PFTauDiscriminantManager::buildMVAComputerLink(std::vector<PhysicsTools::Variable::Value>& toFill)
{
   for(discriminantHolder::iterator iVariable  = myDiscriminants_.begin();
                                    iVariable != myDiscriminants_.end();
                                  ++iVariable)
   {
      Discriminant * theDiscriminant = iVariable->second;
      theDiscriminant->fillMVA(toFill);
   }
}

vector<const reco::Candidate*>
PFTauDiscriminantManager::getLeafDaughters(const reco::Candidate* input) 
{
   std::vector<const reco::Candidate*> output;

   //check for validity
   if(!input)   
      return output;

   size_t nDaughters = input->numberOfDaughters();
   if(!nDaughters)      //this is a leaf
      output.push_back(input);
   else                 //recurse down this objects daughters
   {
      for(size_t iDaughter = 0; iDaughter < nDaughters; ++iDaughter)
      {
         std::vector<const reco::Candidate*> leafsOnThisBranch = getLeafDaughters(input->daughter(iDaughter));
         output.insert(output.end(), leafsOnThisBranch.begin(), leafsOnThisBranch.end());
      }
   }
   return output;
}

PFTauDiscriminantManager::~PFTauDiscriminantManager()
{
}



} //end namespace



