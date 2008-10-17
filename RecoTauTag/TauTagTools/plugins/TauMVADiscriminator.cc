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
// $Id: TauMVADiscriminator.cc,v 1.2 2008/10/16 00:59:17 friis Exp $
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
      virtual void beginRun( const edm::Run& run, const edm::EventSetup& );
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      InputTag                  pfTauDecayModeSrc_;
      string                    computerName_;
      DiscriminantList          myDiscriminants_;
      PFTauDiscriminantManager  discriminantManager_;

      std::vector<PhysicsTools::Variable::Value>        mvaComputerInput_;
};

TauMVADiscriminator::TauMVADiscriminator(const edm::ParameterSet& iConfig):
                   pfTauDecayModeSrc_(iConfig.getParameter<InputTag>("pfTauDecayModeSrc")),
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

   size_t numberOfTaus = pfTauDecayModes->size();
   for(size_t iDecayMode = 0; iDecayMode < numberOfTaus; ++iDecayMode)
   {
      const PFTauDecayMode& theTauDecayMode = pfTauDecayModes->value(iDecayMode);

      //sets the current tau decay mode as the active object
      discriminantManager_.setEventData(theTauDecayMode, iEvent);
      //applies associated discriminants (see ctor) and constructs the appropriate MVA framework input
      discriminantManager_.buildMVAComputerLink(mvaComputerInput_);
      double output = mvaComputer.eval(mvaComputerInput_);
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
