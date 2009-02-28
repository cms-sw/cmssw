// -*- C++ -*-
//
// Package:    TauMVADiscriminator
// Class:      TauMVADiscriminator
// 
/**\class TauMVADiscriminator TauMVADiscriminator.cc RecoTauTag/TauTagTools/src/TauMVADiscriminator.cc

 Description: Produces a PFTauDiscriminator mapping MVA outputs to PFTau objects
              Requires an AssociationVector of PFTauDecayModes->PFTau objects 
              See RecoTauTag/RecoTau/src/PFTauDecayModeDeterminator.cc

*/
//
// Original Author:  Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)
//         Created:  Fri Aug 15 11:22:14 PDT 2008
// $Id: TauMVADiscriminator.cc,v 1.8 2009/01/27 23:18:35 friis Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTauTag/TauTagTools/interface/Discriminants.h"

#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "RecoTauTag/TauTagTools/interface/DiscriminantList.h"
#include "RecoTauTag/TauTagTools/interface/PFTauDiscriminantManager.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "RecoTauTag/TauTagTools/interface/TauMVADBConfiguration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerRecord.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

//
// class decleration
//
using namespace std;
using namespace edm;
using namespace PFTauDiscriminants;

class TauMVADiscriminator : public edm::EDProducer {
   public:
      explicit TauMVADiscriminator(const edm::ParameterSet&);
      ~TauMVADiscriminator();

      struct  MVAComputerFromDB {
         string                     computerName;
         PhysicsTools::MVAComputer* computer;
         double                     userCut; 
      };

      typedef vector<MVAComputerFromDB>    MVAList;
      typedef map<int, MVAList::iterator> DecayModeToMVAMap;

   private:
      typedef vector<Handle<PFTauDiscriminator> > DiscriminantHandleList;
      typedef vector<Handle<PFTauDiscriminator> >::const_iterator DiscriminantHandleIterator;

      virtual void beginRun( const edm::Run& run, const edm::EventSetup& );
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      InputTag                  pfTauDecayModeSrc_;
      bool                      applyCut_;         //Specify whether to output the MVA value, or whether to use 
                                                   // the cuts specified in the DecayMode VPSet specified in the cfg file 
      std::vector<InputTag>     preDiscriminants_; //These must pass for the MVA value to be computed
      double                    failValue_;        //Specifies discriminant output when the object fails one of the preDiscriminants
      DecayModeToMVAMap         computerMap_;      //Maps decay mode to MVA implementation
      MVAList                   computers_;
      DiscriminantList          myDiscriminants_;  // collection of functions to compute the discriminants
      PFTauDiscriminantManager  discriminantManager_;

      std::vector<PhysicsTools::Variable::Value>        mvaComputerInput_;
};

TauMVADiscriminator::TauMVADiscriminator(const edm::ParameterSet& iConfig):
                   pfTauDecayModeSrc_(iConfig.getParameter<InputTag>("pfTauDecayModeSrc")),
                   applyCut_(iConfig.getParameter<bool>("MakeBinaryDecision")),
                   preDiscriminants_(iConfig.getParameter<std::vector<InputTag> >("preDiscriminants")),
                   failValue_(iConfig.getParameter<double>("prefailValue"))
{
   produces<PFTauDiscriminator>(); //define product

   //get the computer/decay mode map
   vector<ParameterSet> decayModeMap = iConfig.getParameter<vector<ParameterSet> >("computers");
   computers_.reserve(decayModeMap.size());
   for(vector<ParameterSet>::const_iterator iComputer  = decayModeMap.begin();
                                            iComputer != decayModeMap.end();
                                          ++iComputer)
   {
      MVAComputerFromDB toInsert;
      toInsert.computerName = iComputer->getParameter<string>("computerName");
      toInsert.userCut      = iComputer->getParameter<double>("cut");
      toInsert.computer     = NULL;
      MVAList::iterator computerJustAdded = computers_.insert(computers_.end(), toInsert); //add this computer to the end of the list

      //populate the map
      vector<int> associatedDecayModes = iComputer->getParameter<vector<int> >("decayModeIndices");
      for(vector<int>::const_iterator iDecayMode  = associatedDecayModes.begin();
                                      iDecayMode != associatedDecayModes.end();
                                    ++iDecayMode)
      {
         //map this integer specifying the decay mode to the MVA comptuer we just added to the list
         pair<DecayModeToMVAMap::iterator, bool> insertResult = computerMap_.insert(make_pair(*iDecayMode, computerJustAdded));

         //make sure we aren't double mapping a decay mode
         if(insertResult.second == false) { //indicates that the current key (decaymode) has already been entered!
            throw cms::Exception("TauMVADiscriminator::ctor") << "A tau decay mode: " << *iDecayMode << " has been mapped to two different MVA implementations, "
                                                              << insertResult.first->second->computerName << " and " << toInsert.computerName 
                                                              << ". Please check the appropriate cfi file." << std::endl;
         }
      }
   }

   for(DiscriminantList::const_iterator aDiscriminant  = myDiscriminants_.begin();
                                        aDiscriminant != myDiscriminants_.end();
                                      ++aDiscriminant)
   {
      //load the discriminants into the discriminant manager
      discriminantManager_.addDiscriminant(*aDiscriminant);
   }
   
}

TauMVADiscriminator::~TauMVADiscriminator()
{
}

// ------------ method called to produce the data  ------------
void
TauMVADiscriminator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;

   Handle<PFTauDecayModeAssociation> pfTauDecayModes;
   iEvent.getByLabel(pfTauDecayModeSrc_, pfTauDecayModes);

   //initialize discriminant vector w/ the RefProd of the tau collection
   auto_ptr<PFTauDiscriminator> outputProduct(new PFTauDiscriminator(pfTauDecayModes->keyProduct()));

   //get appropriate MVA setup (specified in CFG file)
   //we do this on each produce as the crossing an IOV boundary could change the appropriate record
   //TODO: move this to beginRun(...)?
   ESHandle<PhysicsTools::Calibration::MVAComputerContainer> mvaHandle;
   iSetup.get<TauMVAFrameworkDBRcd>().get(mvaHandle);
   for(MVAList::iterator iMVAComputer  = computers_.begin();
                         iMVAComputer != computers_.end();
                       ++iMVAComputer)
   {
      //refresh the MVA computers
      if(iMVAComputer->computer) { //if is non-NULL
         delete iMVAComputer->computer;
      }
      string nameToGet = iMVAComputer->computerName;
      iMVAComputer->computer = new PhysicsTools::MVAComputer(&mvaHandle.product()->find(nameToGet));
   }

   DiscriminantHandleList                    otherDiscriminants;
   for(std::vector<InputTag>::const_iterator iDiscriminant  = preDiscriminants_.begin();
                                             iDiscriminant != preDiscriminants_.end();
                                           ++iDiscriminant)
   {
      Handle<PFTauDiscriminator> tempDiscriminantHandle;
      iEvent.getByLabel(*iDiscriminant, tempDiscriminantHandle);
      otherDiscriminants.push_back(tempDiscriminantHandle);
   }
                                             
   size_t numberOfTaus = pfTauDecayModes->size();
   for(size_t iDecayMode = 0; iDecayMode < numberOfTaus; ++iDecayMode)
   {
      double output = failValue_;

      // Check if this tau fails one of the specified discriminants
      // This is needed as applying these discriminants on a tau w/o a 
      // lead track doesn't do much good
      bool passesPreDiscriminant = true;

      for(DiscriminantHandleIterator iDiscriminant  = otherDiscriminants.begin();
                                     iDiscriminant != otherDiscriminants.end();
                                   ++iDiscriminant)
      {
         float thisDiscriminant = (*iDiscriminant)->value(iDecayMode);
         if (thisDiscriminant < 0.5)
         {
            passesPreDiscriminant = false;
            break;
         }
      }

      if (passesPreDiscriminant)
      {
         mvaComputerInput_.clear();
         const PFTauDecayMode& theTauDecayMode = pfTauDecayModes->value(iDecayMode);
         //get appropriate MVA computer
         int decayMode = theTauDecayMode.getDecayMode();
         DecayModeToMVAMap::iterator iterToComputer = computerMap_.find(decayMode);
         if(iterToComputer != computerMap_.end()) //if we don't have a MVA mapped to this decay mode, skip it.
         {
            PhysicsTools::MVAComputer* mvaComputer = iterToComputer->second->computer;

            //sets the current tau decay mode as the active object
            discriminantManager_.setEventData(theTauDecayMode, iEvent);
            //applies associated discriminants (see ctor) and constructs the appropriate MVA framework input
            discriminantManager_.buildMVAComputerLink(mvaComputerInput_);
            output = mvaComputer->eval(mvaComputerInput_);
            if (applyCut_)
            {
               //If the user desires a yes or no decision, 
               // use the supplied cut to make a decision
               if (output > iterToComputer->second->userCut) 
                  output = 1.0;
               else 
                  output = 0.0;
            }
         }
      }

      outputProduct->setValue(iDecayMode, output);
   }
   iEvent.put(outputProduct);
}
void 
TauMVADiscriminator::beginRun( const edm::Run& run, const edm::EventSetup& iSetup)
{
}

// ------------ method called once each job just before starting event loop  ------------
void 
TauMVADiscriminator::beginJob(const edm::EventSetup& iSetup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TauMVADiscriminator::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TauMVADiscriminator);
