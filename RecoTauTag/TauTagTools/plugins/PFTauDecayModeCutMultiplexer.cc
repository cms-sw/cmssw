// -*- C++ -*-
//
// Package:    PFTauDecayModeCutMultiplexer
// Class:      PFTauDecayModeCutMultiplexer
// 
/*

 Description: Applies a different cut to a PFTauDiscriminator, depending on the 
              the reconstructed DecayMode (stored by RecoTauTag/RecoTau/PFTauDecayModeIndexProducer) 
              in PFTauDiscriminator form.

              Produces a PFTauDiscriminator output with a binary (0 or 1) output.

              Cuts are specified in the decay mode PSets, which map the cuts 
              to collections of decay mode indices.  These decay mode PSets are defined
              in the same manner as TaNC MVA computers (which also map to specific decay modes)


*/
//
// Original Author:  Evan K. Friis, UC Davis (friis@physics.ucdavis.edu)
//         Created:  Thurs, April 16, 2009
// $Id: PFTauDecayModeCutMultiplexer.cc,v 1.0 $
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

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

//
// class decleration
//
using namespace std;
using namespace edm;
using namespace reco;

class PFTauDecayModeCutMultiplexer : public edm::EDProducer {
   public:
      explicit PFTauDecayModeCutMultiplexer(const edm::ParameterSet&);
      ~PFTauDecayModeCutMultiplexer();

      struct  ComputerAndCut {
         string                                 computerName;
         double                                 userCut;
      };

      typedef vector<ComputerAndCut>    CutList;
      typedef map<int, CutList::iterator> DecayModeToCutMap;

   private:
      typedef vector<Handle<PFTauDiscriminator> > DiscriminantHandleList;
      typedef vector<Handle<PFTauDiscriminator> >::const_iterator DiscriminantHandleIterator;

      virtual void beginRun( const edm::Run& run, const edm::EventSetup& );
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      InputTag                  pfTauDecayModeIndexSrc_;
      InputTag                  discriminantToMultiplex_;
      std::vector<InputTag>     preDiscriminants_; //These must pass for the MVA value to be computed
      DecayModeToCutMap         computerMap_;      //Maps decay mode to MVA implementation
      CutList                   computers_;
};

PFTauDecayModeCutMultiplexer::PFTauDecayModeCutMultiplexer(const edm::ParameterSet& iConfig):
                   pfTauDecayModeIndexSrc_(iConfig.getParameter<InputTag>("PFTauDecayModeSrc")),
                   discriminantToMultiplex_(iConfig.getParameter<InputTag>("PFTauDiscriminantToMultiplex")),
                   preDiscriminants_(iConfig.getParameter<std::vector<InputTag> >("preDiscriminants"))
{
   produces<PFTauDiscriminator>(); //define product

   //get the computer/decay mode map
   vector<ParameterSet> decayModeMap = iConfig.getParameter<vector<ParameterSet> >("computers");
   computers_.reserve(decayModeMap.size());
   for(vector<ParameterSet>::const_iterator iComputer  = decayModeMap.begin();
                                            iComputer != decayModeMap.end();
                                          ++iComputer)
   {
      ComputerAndCut toInsert;
      toInsert.computerName = iComputer->getParameter<string>("computerName");
      toInsert.userCut      = iComputer->getParameter<double>("cut");
      CutList::iterator computerJustAdded = computers_.insert(computers_.end(), toInsert); //add this computer to the end of the list

      //populate the map
      vector<int> associatedDecayModes = iComputer->getParameter<vector<int> >("decayModeIndices");
      for(vector<int>::const_iterator iDecayMode  = associatedDecayModes.begin();
                                      iDecayMode != associatedDecayModes.end();
                                    ++iDecayMode)
      {
         //map this integer specifying the decay mode to the MVA comptuer we just added to the list
         pair<DecayModeToCutMap::iterator, bool> insertResult = computerMap_.insert(make_pair(*iDecayMode, computerJustAdded));

         //make sure we aren't double mapping a decay mode
         if(insertResult.second == false) { //indicates that the current key (decaymode) has already been entered!
            throw cms::Exception("PFTauDecayModeCutMultiplexer::ctor") << "A tau decay mode: " << *iDecayMode << " has been mapped to two different MVA implementations, "
                                                              << insertResult.first->second->computerName << " and " << toInsert.computerName 
                                                              << ". Please check the appropriate cfi file." << std::endl;
         }
      }
   }
}

PFTauDecayModeCutMultiplexer::~PFTauDecayModeCutMultiplexer()
{
   //do nothing
}

// ------------ method called to produce the data  ------------
void
PFTauDecayModeCutMultiplexer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace std;
   using namespace reco;

   Handle<PFTauDiscriminator> pfTauDecayModeIndices;
   iEvent.getByLabel(pfTauDecayModeIndexSrc_, pfTauDecayModeIndices);

   Handle<PFTauDiscriminator> targetDiscriminant;
   iEvent.getByLabel(discriminantToMultiplex_, targetDiscriminant);

   //initialize discriminant vector w/ the RefProd of the tau collection
   auto_ptr<PFTauDiscriminator> outputProduct(new PFTauDiscriminator(pfTauDecayModeIndices->keyProduct()));

   // Get the prediscriminants that must always be satisfied
   DiscriminantHandleList                    otherDiscriminants;
   for(std::vector<InputTag>::const_iterator iDiscriminant  = preDiscriminants_.begin();
                                             iDiscriminant != preDiscriminants_.end();
                                           ++iDiscriminant)
   {
      Handle<PFTauDiscriminator> tempDiscriminantHandle;
      iEvent.getByLabel(*iDiscriminant, tempDiscriminantHandle);
      otherDiscriminants.push_back(tempDiscriminantHandle);
   }
                                             
   size_t numberOfTaus = pfTauDecayModeIndices->size();
   for(size_t iDecayMode = 0; iDecayMode < numberOfTaus; ++iDecayMode)
   {
      double output = 0.;

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
         int decayMode          = lrint(pfTauDecayModeIndices->value(iDecayMode)); //convert to int
         float valueToMultiplex = targetDiscriminant->value(iDecayMode);
         // Get correct cut
         DecayModeToCutMap::iterator iterToComputer = computerMap_.find(decayMode);
         if(iterToComputer != computerMap_.end()) //if we don't have a MVA mapped to this decay mode, skip it, it fails.
         {
            // use the supplied cut to make a decision
            if (valueToMultiplex > iterToComputer->second->userCut) 
               output = 1.0;
            else 
               output = 0.0;
         }
      }

      outputProduct->setValue(iDecayMode, output);
   }
   iEvent.put(outputProduct);
}
void 
PFTauDecayModeCutMultiplexer::beginRun( const edm::Run& run, const edm::EventSetup& iSetup)
{
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFTauDecayModeCutMultiplexer::beginJob(const edm::EventSetup& iSetup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFTauDecayModeCutMultiplexer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFTauDecayModeCutMultiplexer);
