// -*- C++ -*-
//
// Package:    HadronAndPartonSelector
// Class:      HadronAndPartonSelector
//
/**\class HadronAndPartonSelector HadronAndPartonSelector.cc PhysicsTools/JetMCAlgos/plugins/HadronAndPartonSelector.cc
 * \brief Selects hadrons and partons from a collection of GenParticles
 *
 * This producer selects hadrons and partons from a collection of GenParticles and stores vectors of EDM references
 * to these particles in the event. The following hadrons are selected:
 *
 * - b hadrons that do not have other b hadrons as daughters
 * - c hadrons that do not have other c hadrons as daughters or a b hadron as mother
 *
 * The parton selection is generator-specific and is described in each of the parton selectors individually.
 * The producer attempts to automatically determine what generator was used to hadronize events in order to determine
 * what parton selection mode to use. It is also possible to enforce any of the supported parton selection modes.
 *
 * The selected hadrons and partons are finally used by the JetFlavourClustering producer to determine the jet flavour.
 */
//
// Original Author:  Dinko Ferencek
//         Created:  Tue Nov  5 22:43:43 CET 2013
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

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "FWCore/Common/interface/Provenance.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "PhysicsTools/JetMCUtils/interface/CandMCTag.h"
#include "PhysicsTools/JetMCUtils/interface/JetMCTag.h"
#include "PhysicsTools/JetMCAlgos/interface/BasePartonSelector.h"
#include "PhysicsTools/JetMCAlgos/interface/PythiaPartonSelector.h"

//
// constants, enums and typedefs
//
typedef boost::shared_ptr<BasePartonSelector>  PartonSelectorPtr;

//
// class declaration
//

class HadronAndPartonSelector : public edm::EDProducer {
   public:
      explicit HadronAndPartonSelector(const edm::ParameterSet&);
      ~HadronAndPartonSelector();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run&, edm::EventSetup const&);
      virtual void endRun(edm::Run&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------
      const edm::EDGetTokenT<GenEventInfoProduct>         srcToken_;        // To get handronizer module type
      const edm::EDGetTokenT<reco::GenParticleCollection> particlesToken_;  // Input GenParticle collection

      std::string         partonMode_; // Parton selection mode
      PartonSelectorPtr   partonSelector_;
};

//
// static data member definitions
//

//
// constructors and destructor
//
HadronAndPartonSelector::HadronAndPartonSelector(const edm::ParameterSet& iConfig) :

  srcToken_(mayConsume<GenEventInfoProduct>( iConfig.getParameter<edm::InputTag>("src") )),
  particlesToken_(consumes<reco::GenParticleCollection>( iConfig.getParameter<edm::InputTag>("particles") )),
  partonMode_(iConfig.getParameter<std::string>("partonMode"))

{
   //register your products
   produces<reco::GenParticleRefVector>( "bHadrons" );
   produces<reco::GenParticleRefVector>( "cHadrons" );
   produces<reco::GenParticleRefVector>( "partons" );
}


HadronAndPartonSelector::~HadronAndPartonSelector()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HadronAndPartonSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // determine hadronizer type (done only once per job)
   if( partonMode_=="Auto" )
   {
     edm::Handle<GenEventInfoProduct> genEvtInfoProduct;
     iEvent.getByToken(srcToken_, genEvtInfoProduct);

     std::string moduleName = "";
     const edm::Provenance& prov = iEvent.getProvenance(genEvtInfoProduct.id());
     if( genEvtInfoProduct.isValid() )
       moduleName = edm::moduleName(prov);

     if( moduleName.find("Pythia6")!=std::string::npos )
       partonMode_="Pythia6";
     else if( moduleName.find("Pythia8")!=std::string::npos )
       partonMode_="Pythia8";
     else
       partonMode_="Undefined";
   }

   // set the parton selection mode
   if ( partonMode_=="Undefined" )
     edm::LogWarning("UndefinedPartonMode") << "Could not automatically determine the hadronizer type and set the correct parton selection mode. Parton-based jet flavour will not be defined.";
   else if ( partonMode_=="Pythia6" || partonMode_=="Pythia8" )
     partonSelector_ = PartonSelectorPtr( new PythiaPartonSelector() );
   else
     //throw cms::Exception("InvalidPartonMode") <<"Parton selection mode is invalid: " << partonMode_ << ", use Auto | Pythia6 | Pythia8 | Herwig6 | Herwig++ | Sherpa" << std::endl;
     throw cms::Exception("InvalidPartonMode") <<"Parton selection mode is invalid: " << partonMode_ << ", use Auto | Pythia6 | Pythia8" << std::endl;


   edm::Handle<reco::GenParticleCollection> particles;
   iEvent.getByToken(particlesToken_, particles);

   std::auto_ptr<reco::GenParticleRefVector> bHadrons ( new reco::GenParticleRefVector );
   std::auto_ptr<reco::GenParticleRefVector> cHadrons ( new reco::GenParticleRefVector );
   std::auto_ptr<reco::GenParticleRefVector> partons  ( new reco::GenParticleRefVector );

   // loop over particles and select b and c hadrons
   for(reco::GenParticleCollection::const_iterator it = particles->begin(); it != particles->end(); ++it)
   {
     // if b hadron
     if( CandMCTagUtils::hasBottom( *it ) )
     {
       // check if any of the daughters is also a b hadron
       bool hasbHadronDaughter = false;
       for(size_t i=0; i < it->numberOfDaughters(); ++i)
       {
         if( CandMCTagUtils::hasBottom( *(it->daughter(i)) ) ) { hasbHadronDaughter = true; break; }
       }
       if( hasbHadronDaughter ) continue; // skip excited b hadrons that have other b hadrons as daughters

       bHadrons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
     }

     // if c hadron
     if( CandMCTagUtils::hasCharm( *it ) )
     {
       // check if any of the mothers is a b hadron
       if( JetMCTagUtils::decayFromBHadron( *it ) ) continue; // skip c hadrons that have a b hadron as mother

       // check if any of the daughters is also a c hadron
       bool hascHadronDaughter = false;
       for(size_t i=0; i < it->numberOfDaughters(); ++i)
       {
         if( CandMCTagUtils::hasCharm( *(it->daughter(i)) ) ) { hascHadronDaughter = true; break; }
       }
       if( hascHadronDaughter ) continue; // skip excited c hadrons that have other c hadrons as daughters

       cHadrons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
     }
   }

   // select partons
   if ( partonMode_!="Undefined" )
     partonSelector_->run(particles,partons);

   iEvent.put( bHadrons, "bHadrons" );
   iEvent.put( cHadrons, "cHadrons" );
   iEvent.put( partons,  "partons" );
}

// ------------ method called once each job just before starting event loop  ------------
void
HadronAndPartonSelector::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
HadronAndPartonSelector::endJob() {
}

// ------------ method called when starting to processes a run  ------------
void
HadronAndPartonSelector::beginRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a run  ------------
void
HadronAndPartonSelector::endRun(edm::Run&, edm::EventSetup const&)
{
}

// ------------ method called when starting to processes a luminosity block  ------------
void
HadronAndPartonSelector::beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void
HadronAndPartonSelector::endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HadronAndPartonSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HadronAndPartonSelector);
