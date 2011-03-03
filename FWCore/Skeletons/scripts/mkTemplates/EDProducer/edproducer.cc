// -*- C++ -*-
//
// Package:    prodname
// Class:      prodname
// 
/**\class prodname prodname.cc skelsubsys/prodname/src/prodname.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  John Doe
//         Created:  day-mon-xx
// RCS(Id)
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

@example_myparticle #include "DataFormats/MuonReco/interface/Muon.h"
@example_myparticle #include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
@example_myparticle #include "DataFormats/Candidate/interface/Particle.h"
@example_myparticle #include "FWCore/MessageLogger/interface/MessageLogger.h"
@example_myparticle #include "FWCore/Utilities/interface/InputTag.h"

//
// class declaration
//

class prodname : public edm::EDProducer {
   public:
      explicit prodname(const edm::ParameterSet&);
      ~prodname();

      static void fillDescriptions(ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual void beginRun(Run&, EventSetup const&);
      virtual void endRun(Run&, EventSetup const&):
      virtual void beginLuminosityBlock(LuminosityBlock&, EventSetup const&);
      virtual void endLuminosityBlock(LuminosityBlock&, EventSetup const&);

      // ----------member data ---------------------------
@example_myparticle       edm::InputTag muonTags_; 
@example_myparticle       edm::InputTag electronTags_;
};

//
// constants, enums and typedefs
//

@example_myparticle       // define container that will be booked into event
@example_myparticle       typedef std::vector<reco::Particle> MyParticleCollection;

//
// static data member definitions
//

//
// constructors and destructor
//
prodname::prodname(const edm::ParameterSet& iConfig)
@example_myparticle :
@example_myparticle   muonTags_( iConfig.getParameter<edm::InputTag>( "muons" )),
@example_myparticle   electronTags_( iConfig.getParameter<edm::InputTag>( "electrons" ))
{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
 
   //if you want to put into the Run
   produces<ExampleData2,InRun>();
*/
@example_myparticle   produces<MyParticleCollection>( "particles" );
   //now do what ever other initialization is needed
  
}


prodname::~prodname()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
prodname::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
@example_myparticle    using namespace reco;
@example_myparticle    using namespace std;
/* This is an event example
   //Read 'ExampleData' from the Event
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);

   //Use the ExampleData to create an ExampleData2 which 
   // is put into the Event
   std::auto_ptr<ExampleData2> pOut(new ExampleData2(*pIn));
   iEvent.put(pOut);
*/

/* this is an EventSetup example
   //Read SetupData from the SetupRecord in the EventSetup
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
*/
 
@example_myparticle    Handle<MuonCollection> muons;
@example_myparticle    iEvent.getByLabel( muonTags_, muons );
@example_myparticle    
@example_myparticle    Handle<PixelMatchGsfElectronCollection> electrons;
@example_myparticle    iEvent.getByLabel( electronTags_, electrons );
@example_myparticle    
@example_myparticle    // create a new collection of Particle objects
@example_myparticle    auto_ptr<MyParticleCollection> newParticles( new MyParticleCollection );      
@example_myparticle 
@example_myparticle    // if the number of electrons or muons is 4 (or 2 and 2), costruct a new particle
@example_myparticle    if( muons->size() == 4 || electrons->size() == 4 || ( muons->size() == 2 && electrons->size() == 2 ) ) {
@example_myparticle       
@example_myparticle       // sums of momenta and charges will be calculated
@example_myparticle       Particle::LorentzVector totalP4( 0, 0, 0, 0 );
@example_myparticle       Particle::Charge charge( 0 );
@example_myparticle       
@example_myparticle       // loop over muons, sum over p4s and charges. Later same for electrons
@example_myparticle       for( MuonCollection::const_iterator muon = muons->begin(); muon != muons->end(); ++muon ) {
@example_myparticle          totalP4 += muon->p4();
@example_myparticle          charge += muon->charge();
@example_myparticle       }
@example_myparticle       
@example_myparticle       for( PixelMatchGsfElectronCollection::const_iterator electron = electrons->begin(); electron != electrons->end(); ++electron ) {
@example_myparticle          totalP4 += electron->p4(); 
@example_myparticle          charge += electron->charge(); 
@example_myparticle       }
@example_myparticle       
@example_myparticle       // create a particle with momentum and charge from muons and electrons
@example_myparticle       Particle h;
@example_myparticle       h.setP4(totalP4);
@example_myparticle       h.setCharge(charge);
@example_myparticle 
@example_myparticle       // fill the particles into the vector
@example_myparticle       newParticles->push_back( h );      
@example_myparticle    }
@example_myparticle    
@example_myparticle    // save the vector
@example_myparticle    iEvent.put( newParticles, "particles" );
}

// ------------ method called once each job just before starting event loop  ------------
void 
prodname::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
prodname::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void 
prodname::beginRun(Run&, EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void 
prodname::endRun(Run&, EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void 
prodname::beginLuminosityBlock(LuminosityBlock&, EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
prodname::endLuminosityBlock(LuminosityBlock&, EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
prodname::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
@example_myparticle  
@example_myparticle  //Specify that only 'muons' and 'electrons' are allowed
@example_myparticle  //To use, remove the default given above and uncomment below
@example_myparticle  //ParameterSetDescription desc;
@example_myparticle  //desc.add<edm::InputTag>("muons","muons");
@example_myparticle  //desc.add<edm::InputTag>("electrons","pixelMatchGsfElectrons");
@example_myparticle  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(prodname);
