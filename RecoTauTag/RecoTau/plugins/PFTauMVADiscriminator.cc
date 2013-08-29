// -*- C++ -*-
//
// Package:    PFTauMVADiscriminator
// Class:      PFTauMVADiscriminator
// 
/**\class PFTauMVADiscriminator PFTauMVADiscriminator.cc RecoTauTag/RecoTau/plugins/PFTauMVADiscriminator.cc

 Description: Produces a PFTauDiscriminator mapping MVA outputs to PFTau objects
              Requires an AssociationVector of PFTauDecayModes->PFTau objects 
              See RecoTauTag/RecoTau/src/PFTauDecayModeDeterminator.cc

*/
//
// Original Author:  Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)
//         Created:  Fri Aug 15 11:22:14 PDT 2008
//
//

#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"

// Tau discriminant computation
#include "RecoTauTag/TauTagTools/interface/PFTauDiscriminantManager.h"
#include "RecoTauTag/TauTagTools/interface/Discriminants.h"
#include "RecoTauTag/TauTagTools/interface/DiscriminantList.h"

// PFTauDecayMode data formats
#include "DataFormats/TauReco/interface/PFTauDecayMode.h"
#include "DataFormats/TauReco/interface/PFTauDecayModeAssociation.h"

// DB access
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "RecoTauTag/TauTagTools/interface/TauMVADBConfiguration.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerRecord.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerCache.h"

using namespace PFTauDiscriminants;
using namespace reco;

class PFTauMVADiscriminator : public PFTauDiscriminationProducerBase {
   public:
      explicit PFTauMVADiscriminator(const edm::ParameterSet&);
      ~PFTauMVADiscriminator();

      struct  MVAComputerFromDB {
         std::string                                 computerName;
         PhysicsTools::MVAComputerCache*        computer;
         double                                 userCut;
      };

      typedef std::vector<MVAComputerFromDB>    MVAList;
      typedef std::map<int, MVAList::iterator> DecayModeToMVAMap;

      void beginEvent(const edm::Event&, const edm::EventSetup&); // called at the beginning of each event
      double discriminate(const PFTauRef&);             // called on every tau in input collection

   private:
      edm::InputTag                  pfTauDecayModeSrc_;
      bool                      remapOutput_;      // TMVA defaults output to (-1, 1).  Option to remap to (0, 1)
      bool                      applyCut_;         //Specify whether to output the MVA value, or whether to use 
                                                   // the cuts specified in the DecayMode VPSet specified in the cfg file 
      DecayModeToMVAMap         computerMap_;      //Maps decay mode to MVA implementation
      MVAList                   computers_;
      std::string               dbLabel_;
      DiscriminantList          myDiscriminants_;  // collection of functions to compute the discriminants
      PFTauDiscriminantManager  discriminantManager_;

      edm::Handle<PFTauDecayModeAssociation> pfTauDecayModes; // edm::Handle to PFTauDecayModes for current event

      std::vector<PhysicsTools::Variable::Value>        mvaComputerInput_;
};

PFTauMVADiscriminator::PFTauMVADiscriminator(const edm::ParameterSet& iConfig):PFTauDiscriminationProducerBase(iConfig)
{
   pfTauDecayModeSrc_        = iConfig.getParameter<edm::InputTag>("pfTauDecayModeSrc");
   remapOutput_              = iConfig.getParameter<bool>("RemapOutput");
   applyCut_                 = iConfig.getParameter<bool>("MakeBinaryDecision");
   prediscriminantFailValue_ = iConfig.getParameter<double>("prefailValue"); //defined in base class
   dbLabel_                  = iConfig.getParameter<std::string>("dbLabel");

   // build the decaymode->computer map
   std::vector<edm::ParameterSet> decayModeMap = iConfig.getParameter<std::vector<edm::ParameterSet> >("computers");
   computers_.reserve(decayModeMap.size());
   for(std::vector<edm::ParameterSet>::const_iterator iComputer  = decayModeMap.begin(); iComputer != decayModeMap.end(); ++iComputer)
   {
      MVAComputerFromDB toInsert;
      toInsert.computerName = iComputer->getParameter<std::string>("computerName");
      toInsert.userCut      = iComputer->getParameter<double>("cut");
      toInsert.computer     = new PhysicsTools::MVAComputerCache();
      MVAList::iterator computerJustAdded = computers_.insert(computers_.end(), toInsert); //add this computer to the end of the list

      //populate the map
      std::vector<int> associatedDecayModes = iComputer->getParameter<std::vector<int> >("decayModeIndices");
      for(std::vector<int>::const_iterator iDecayMode  = associatedDecayModes.begin();
                                      iDecayMode != associatedDecayModes.end();
                                    ++iDecayMode)
      {
         //map this integer specifying the decay mode to the MVA comptuer we just added to the list
         std::pair<DecayModeToMVAMap::iterator, bool> insertResult = computerMap_.insert(std::make_pair(*iDecayMode, computerJustAdded));

         //make sure we aren't double mapping a decay mode
         if(insertResult.second == false) { //indicates that the current key (decaymode) has already been entered!
            throw cms::Exception("PFTauMVADiscriminator::ctor") << "A tau decay mode: " << *iDecayMode << " has been mapped to two different MVA implementations, "
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

PFTauMVADiscriminator::~PFTauMVADiscriminator()
{
   for(MVAList::iterator iMVAComputer  = computers_.begin(); iMVAComputer != computers_.end(); ++iMVAComputer)
   {
      delete iMVAComputer->computer;
   }
}

// ------------ method called at the beginning of every event by base class
void PFTauMVADiscriminator::beginEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // load the PFTauDecayModes
   iEvent.getByLabel(pfTauDecayModeSrc_, pfTauDecayModes);

   // expose the event to the PFTau discriminant quantity computers
   discriminantManager_.setEvent(iEvent, 1.0); //event weight = 1 

   // Refresh MVAs
   //we do this on each event as the crossing an IOV boundary could change the appropriate record
   for(MVAList::iterator iMVAComputer  = computers_.begin();
                         iMVAComputer != computers_.end();
                       ++iMVAComputer)
   {
      std::string nameToGet = iMVAComputer->computerName;
      iMVAComputer->computer->update<TauMVAFrameworkDBRcd>(dbLabel_.c_str(), iSetup, nameToGet.c_str());
   } 
}

double PFTauMVADiscriminator::discriminate(const PFTauRef& pfTau)
{
   double result = prediscriminantFailValue_;

   mvaComputerInput_.clear();
   const PFTauDecayMode& theTauDecayMode = (*pfTauDecayModes)[pfTau]; // get PFTauDecayMode associated to this PFTau


   //get appropriate MVA computer
   int decayMode = theTauDecayMode.getDecayMode();
   DecayModeToMVAMap::iterator iterToComputer = computerMap_.find(decayMode);

   if(iterToComputer != computerMap_.end()) //if we don't have a MVA mapped to this decay mode, skip it.
   {
      const PhysicsTools::MVAComputerCache* mvaComputer = iterToComputer->second->computer;
      if ( (*mvaComputer) ) 
      {
         //sets the current tau decay mode as the active object
         discriminantManager_.setTau(theTauDecayMode);
         //applies associated discriminants (see ctor) and constructs the appropriate MVA framework input
         discriminantManager_.buildMVAComputerLink(mvaComputerInput_);
         result = (*mvaComputer)->eval(mvaComputerInput_);
      } 
      else 
      {
         edm::LogWarning("PFTauMVADiscriminator") << "Warning: got a null pointer to MVA computer in conditions database"
            << " for decay mode: " << decayMode << ", expected MVA computer name: " 
            << iterToComputer->second->computerName;
      }

      if (remapOutput_) // TMVA maps result to [-1, 1].  Remap, if desired, to [0, 1]
      {
         if      (result >  1) result = 1.;
         else if (result < -1) result = 0.;
         else { 
            result += 1.;
            result /= 2.;
         }
      }
      if (applyCut_)
      {
         //If the user desires a yes or no decision, 
         // use the supplied cut to make a decision
         if (result > iterToComputer->second->userCut) 
            result = 1.0;
         else 
            result = 0.0;
      }
   }

   return result;
}


//define this as a plug-in
DEFINE_FWK_MODULE(PFTauMVADiscriminator);
