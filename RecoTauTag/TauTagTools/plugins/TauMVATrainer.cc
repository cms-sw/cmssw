// -*- C++ -*-
//
// Package:    TauMVATrainer
// Class:      TauMVATrainer
// 
/**\class TauMVATrainer TauMVATrainer.cc RecoTauTag/TauTagTools/src/TauMVATrainer.cc

Description: Generates ROOT trees used to train PhysicsTools::MVAComputers

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Evan K.Friis, UC Davis  (friis@physics.ucdavis.edu)
//         Created:  Fri Aug 15 11:22:14 PDT 2008
// $Id: TauMVATrainer.cc,v 1.7 2010/10/19 20:22:27 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TTree.h"
#include "TFile.h"
#include "RecoTauTag/TauTagTools/interface/TauDecayModeTruthMatcher.h"
#include "RecoTauTag/TauTagTools/interface/DiscriminantList.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"

//
// class decleration
//
using namespace std;
using namespace edm;
using namespace reco;
using namespace PFTauDiscriminants;

class TauMVATrainer : public edm::EDAnalyzer {
   public:
      
      struct tauMatchingInfoHolder {
         InputTag                               truthToRecoTauMatchingTag;         
         edm::Handle<PFTauDecayModeMatchMap>         truthToRecoTauMatchingHandle;
         InputTag                               decayModeToRecoTauAssociationTag;
         edm::Handle<PFTauDecayModeAssociation>      decayModeToRecoTauAssociationHandle;
         TTree*                                 associatedTTree;
      };

      typedef std::vector<tauMatchingInfoHolder> tauMatchingInfos;
      typedef std::vector<std::pair<TTree*, const PFTauDecayModeMatchMap*> > treeToMatchTuple;

      explicit TauMVATrainer(const edm::ParameterSet&);
      ~TauMVATrainer();
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

   private:
      // ----------member data ---------------------------
      InputTag                  mcTruthSource_;
      //vector<InputTag>          matchingSources_;
      std::vector<ParameterSet>      matchingSources_;
      std::vector<tauMatchingInfoHolder> matchingInfo_;
      bool                      iAmSignal_;

      uint32_t                  maxTracks_;     //any objects w/ nTracks > will be automatically flagged as background
      uint32_t                  maxPiZeroes_;   //any objects w/ nPiZeros > will be automatically flagged as background

      std::string               outputRootFileName_;
      std::map<string, TTree*>       myTrainerTrees_;
      TTree*                    theTruthTree_;  //cache this to prevent string lookup
      PFTauDiscriminantManager  discriminantManager_;
      TFile*                    outputFile_;
      DiscriminantList          myDiscriminants_;
};


//
// constructors and destructor
//
TauMVATrainer::TauMVATrainer(const edm::ParameterSet& iConfig):
                   mcTruthSource_(iConfig.getParameter<InputTag>("mcTruthSource")),
                   matchingSources_(iConfig.getParameter<vector<ParameterSet> >("matchingSources")),
                   iAmSignal_(iConfig.getParameter<bool>("iAmSignal")),
                   maxTracks_(iConfig.getParameter<uint32_t>("maxTracks")),
                   maxPiZeroes_(iConfig.getParameter<uint32_t>("maxPiZeroes")),
                   outputRootFileName_(iConfig.getParameter<string>("outputRootFileName"))

{
   outputFile_ = new TFile(outputRootFileName_.c_str(), "RECREATE");
   edm::LogInfo("TauMVATrainer") << "Initializing TauMVATrainer ctor...";
   // set as signal or background
   discriminantManager_.setSignalFlag(iAmSignal_);

   edm::LogInfo("TauMVATrainer") << "Adding discriminants to TauDiscriminantManager...";
   // add the discriminants to the discriminant manager
   for(DiscriminantList::const_iterator aDiscriminant  = myDiscriminants_.begin();
                                        aDiscriminant != myDiscriminants_.end();
                                      ++aDiscriminant)
   {
      discriminantManager_.addDiscriminant(*aDiscriminant);
   }

   //create tree to hold truth variables
   edm::LogInfo("TauMVATrainer") << "Building truth tree...";
   TTree* truthTree = new TTree("truth", "truth");
//   truthTree->SetDebug();
   myTrainerTrees_.insert(make_pair("truth", truthTree));
   theTruthTree_ = truthTree;
   // branch this trees according to the holder variables in the discrimimnant manager
   discriminantManager_.branchTree(truthTree);

   for(std::vector<ParameterSet>::const_iterator iSrc  = matchingSources_.begin();
                                            iSrc != matchingSources_.end();
                                          ++iSrc)
   {
      //create new matching info record
      tauMatchingInfoHolder aMatcher;
      //create a new tree for each input source
      aMatcher.truthToRecoTauMatchingTag        = iSrc->getParameter<InputTag>("truthMatchSource");
      aMatcher.decayModeToRecoTauAssociationTag = iSrc->getParameter<InputTag>("decayModeAssociationSource"); 
      string label = aMatcher.decayModeToRecoTauAssociationTag.label();
      edm::LogInfo("TauMVATrainer") << "Building reco tree w/ label: " << label << "...";
      TTree* newTree = new TTree(label.c_str(),label.c_str());
      discriminantManager_.branchTree(newTree);
      aMatcher.associatedTTree = newTree;
      matchingInfo_.push_back(aMatcher);
      myTrainerTrees_.insert(make_pair(label, newTree));
   }

}


TauMVATrainer::~TauMVATrainer()
{
}


// ------------ method called to produce the data  ------------
void
TauMVATrainer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;


   // get list of MC Truth objects
   edm::Handle<PFTauDecayModeCollection> truthObjects;
   iEvent.getByLabel(mcTruthSource_, truthObjects);

   discriminantManager_.setEvent(iEvent, 1.0); // unit weight for now

   size_t numberOfTruthObjects = truthObjects->size();
   // loop over true MCTaus and find matched reco objects for each producer
   for(size_t iTrueTau = 0; iTrueTau < numberOfTruthObjects; ++iTrueTau)
   {
      PFTauDecayModeRef theTrueTau = PFTauDecayModeRef(truthObjects, iTrueTau);

      // compute quantities for the truth object and fill associated tree
      discriminantManager_.setTau(*theTrueTau);
      theTruthTree_->Fill();

      // loop over the reco object collections
      for(tauMatchingInfos::iterator iMatchingInfo  = matchingInfo_.begin();
                                     iMatchingInfo != matchingInfo_.end();
                                   ++iMatchingInfo)
      {
         //get matching info from event
         edm::Handle<PFTauDecayModeMatchMap>& theMatching = iMatchingInfo->truthToRecoTauMatchingHandle;
         iEvent.getByLabel(iMatchingInfo->truthToRecoTauMatchingTag, theMatching);

         //get PFTau->PFTauDecayMode association from event
         edm::Handle<PFTauDecayModeAssociation>& theDMAssoc = iMatchingInfo->decayModeToRecoTauAssociationHandle;
         iEvent.getByLabel(iMatchingInfo->decayModeToRecoTauAssociationTag, theDMAssoc);

         //get associated ttree
         TTree* treeToFill           = iMatchingInfo->associatedTTree;

         // Retrieves associated PFTau
         PFTauRef          theAssociatedRecoTau   = (*theMatching)[theTrueTau];

         //determine if there is a RECO match and make sure it has at least one charged signal occupant
         bool isNonNull = (theAssociatedRecoTau.isNonnull() && theAssociatedRecoTau->signalPFChargedHadrCands().size());

         // apply discriminants if there is an associated reconstructed object with at least one track
         if(isNonNull)
         {
            // From associated PFTau get the DecayMode reconstruction
            const PFTauDecayMode& theAssociatedDecayMode = (*theDMAssoc)[theAssociatedRecoTau];
            //determine if tau needs a PRE-pass/fail cut
            bool prePass = false;
            bool preFail = false;
            unsigned int numberOfTracks   = theAssociatedDecayMode.chargedPions().numberOfDaughters();
            unsigned int charge           = std::abs(theAssociatedDecayMode.charge());
            unsigned int numberOfPiZeros  = theAssociatedDecayMode.neutralPions().numberOfDaughters();
            unsigned int numberOfOutliers = theAssociatedDecayMode.pfTauRef()->isolationPFCands().size();
            //cut on high multiplicity
            if (numberOfTracks > maxTracks_ || numberOfPiZeros > maxPiZeroes_  || (charge != 1 && numberOfTracks == 3))
               preFail = true;
            //cut on isolated single prong
            else if (numberOfTracks == 1 && numberOfPiZeros == 0 && numberOfOutliers == 0)
            {
               prePass = true;
            }

            discriminantManager_.setTau(theAssociatedDecayMode, prePass, preFail);
         }
         else 
            // if not, set the null flag
            discriminantManager_.setNullResult();
         treeToFill->Fill();
      }
   }
}

// ------------ method called once each job just before starting event loop  ------------
void 
TauMVATrainer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TauMVATrainer::endJob() {
   for(std::map<string, TTree*>::iterator iTree  = myTrainerTrees_.begin();
                                     iTree != myTrainerTrees_.end();
                                   ++iTree)
   {
      const TTree* myTree = iTree->second;
      edm::LogInfo("TauMVATrainer") << "Tree " << myTree->GetName() << " has " << myTree->GetEntries() << " entries.";
   }
   outputFile_->Write();
}

//define this as a plug-in

DEFINE_FWK_MODULE(TauMVATrainer);
//DEFINE_FWK_MODULE(TauMVATrainer);
