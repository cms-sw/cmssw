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
// $Id: TauMVADiscriminator.cc,v 1.4 2008/10/22 00:51:03 friis Exp $
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
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
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

   private:
      typedef vector<Handle<PFTauDiscriminator> > DiscriminantHandleList;
      typedef vector<Handle<PFTauDiscriminator> >::const_iterator DiscriminantHandleIterator;

      virtual void beginRun( const edm::Run& run, const edm::EventSetup& );
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      InputTag                  pfTauDecayModeSrc_;
      std::vector<InputTag>     preDiscriminants_; //these must pass for the MVA value to be computed
      double                    failValue_;        //specifies discriminant output when the object fails one of the preDiscriminants
      string                    computerName_;
      DiscriminantList          myDiscriminants_;
      PFTauDiscriminantManager  discriminantManager_;

      std::vector<PhysicsTools::Variable::Value>        mvaComputerInput_;
};

TauMVADiscriminator::TauMVADiscriminator(const edm::ParameterSet& iConfig):
                   pfTauDecayModeSrc_(iConfig.getParameter<InputTag>("pfTauDecayModeSrc")),
                   preDiscriminants_(iConfig.getParameter<std::vector<InputTag> >("preDiscriminants")),
                   failValue_(iConfig.getParameter<double>("prefailValue")),
                   computerName_(iConfig.getParameter<string>("computerName"))
{
   produces<PFTauDiscriminator>();

   for(DiscriminantList::const_iterator aDiscriminant  = myDiscriminants_.begin();
                                        aDiscriminant != myDiscriminants_.end();
                                      ++aDiscriminant)
   {
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
   ESHandle<PhysicsTools::Calibration::MVAComputerContainer> mvaHandle;
   iSetup.get<BTauGenericMVAJetTagComputerRcd>().get(mvaHandle);
   PhysicsTools::MVAComputer mvaComputer(&mvaHandle.product()->find(computerName_));

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
         //sets the current tau decay mode as the active object
         discriminantManager_.setEventData(theTauDecayMode, iEvent);
         //applies associated discriminants (see ctor) and constructs the appropriate MVA framework input
         discriminantManager_.buildMVAComputerLink(mvaComputerInput_);
         output = mvaComputer.eval(mvaComputerInput_);
#undef EK_MVA_DEBUG
#ifdef EK_MVA_DEBUG
         std::cout << "Passed PreDisc. DecayMode: " << theTauDecayMode.getDecayMode() <<  " Pt " << theTauDecayMode.pt() 
                   << " eta: " << theTauDecayMode.eta() << " neutral Pt: " << theTauDecayMode.neutralPions().pt() << " MVA: " << output << std::endl; 
#endif
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
