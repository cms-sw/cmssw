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
// $Id$
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
#include "RecoTauTag/TauTagTools/interface/Discriminants.h"
#include "RecoTauTag/TauTagTools/interface/PFTauDiscriminantManager.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
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
      vector<Discriminant*>     myDiscriminants_;
      PFTauDiscriminantManager  discriminantManager_;

      std::vector<PhysicsTools::Variable::Value>        mvaComputerInput_;
};

TauMVADiscriminator::TauMVADiscriminator(const edm::ParameterSet& iConfig):
                   pfTauDecayModeSrc_(iConfig.getParameter<InputTag>("pfTauDecayModeSrc"))
{
   produces<PFTauDiscriminator>();

   myDiscriminants_.push_back(new DecayMode());
   myDiscriminants_.push_back(new MainTrackPt());
   myDiscriminants_.push_back(new MainTrackAngle());
   myDiscriminants_.push_back(new TrackPt());
   myDiscriminants_.push_back(new TrackAngle());
   myDiscriminants_.push_back(new PiZeroPt());
   myDiscriminants_.push_back(new PiZeroAngle());
   myDiscriminants_.push_back(new Dalitz());
   myDiscriminants_.push_back(new InvariantMassOfSignal());
   myDiscriminants_.push_back(new InvariantMass());
   myDiscriminants_.push_back(new Pt());
   myDiscriminants_.push_back(new Eta());
   myDiscriminants_.push_back(new OutlierPt());
   myDiscriminants_.push_back(new OutlierAngle());
   myDiscriminants_.push_back(new ChargedOutlierPt());
   myDiscriminants_.push_back(new ChargedOutlierAngle());
   myDiscriminants_.push_back(new NeutralOutlierPt());
   myDiscriminants_.push_back(new NeutralOutlierAngle());
   myDiscriminants_.push_back(new OutlierNCharged());

   for(vector<Discriminant*>::const_iterator aDiscriminant  = myDiscriminants_.begin();
                                             aDiscriminant != myDiscriminants_.end();
                                           ++aDiscriminant)
   {
      discriminantManager_.addDiscriminant(*aDiscriminant);
   }
}

TauMVADiscriminator::~TauMVADiscriminator()
{
   //clean up
   for(vector<Discriminant*>::iterator aDiscriminant  = myDiscriminants_.begin();
                                       aDiscriminant != myDiscriminants_.end();
                                     ++aDiscriminant)
   {
      delete *aDiscriminant;
   }
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
   ESHandle<PhysicsTools::Calibration::MVAComputer> mvaHandle;
   iSetup.get<MVAComputerRecord>().get(mvaHandle);
   PhysicsTools::MVAComputer mvaComputer(mvaHandle.product());

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
