// -*- C++ -*-
//
// Package:    HadronAndPartonSelector
// Class:      HadronAndPartonSelector
//
/**\class HadronAndPartonSelector HadronAndPartonSelector.cc PhysicsTools/JetMCAlgos/plugins/HadronAndPartonSelector.cc
 * \brief Selects hadrons and partons from a collection of GenParticles
 *
 * This producer selects hadrons, partons, and leptons from a collection of GenParticles and stores vectors of EDM references
 * to these particles in the event. The following hadrons are selected:
 *
 * - b hadrons that do not have other b hadrons as daughters
 * 
 * - c hadrons that do not have other c hadrons as daughters
 * 
 * Older Fortran Monte Carlo generators (Pythia6 and Herwig6) follow the HEPEVT [1] particle status code convention while
 * newer C++ Monte Carlo generators (Pythia8, Herwig++, and Sherpa) follow the HepMC [2] particle status code convention.
 * However, both conventions give considerable freedom in defining the status codes of intermediate particle states. Hence,
 * the parton selection is generator-dependent and is described in each of the parton selectors separately.
 * 
 * Using the provenance information of the GenEventInfoProduct, the producer attempts to automatically determine what generator
 * was used to hadronize events and based on that information decides what parton selection mode to use. It is also possible
 * to enforce any of the supported parton selection modes.
 *
 * The selected hadrons and partons are finally used by the JetFlavourClustering producer to determine the jet flavour.
 * 
 * The following leptons are selected:
 * 
 * - status==1 electrons and muons
 * 
 * - status==2 taus
 * 
 * 
 * [1] http://cepa.fnal.gov/psm/stdhep/
 * [2] http://lcgapp.cern.ch/project/simu/HepMC/
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
#include "PhysicsTools/CandUtils/interface/pdgIdUtils.h"
#include "PhysicsTools/JetMCAlgos/interface/BasePartonSelector.h"
#include "PhysicsTools/JetMCAlgos/interface/Pythia6PartonSelector.h"
#include "PhysicsTools/JetMCAlgos/interface/Pythia8PartonSelector.h"
#include "PhysicsTools/JetMCAlgos/interface/Herwig6PartonSelector.h"
#include "PhysicsTools/JetMCAlgos/interface/HerwigppPartonSelector.h"
#include "PhysicsTools/JetMCAlgos/interface/SherpaPartonSelector.h"

//
// constants, enums and typedefs
//
typedef boost::shared_ptr<BasePartonSelector> PartonSelectorPtr;

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
   produces<reco::GenParticleRefVector>( "leptons" );
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
     else if( moduleName.find("Herwig6")!=std::string::npos )
       partonMode_="Herwig6";
     else if( moduleName.find("ThePEG")!=std::string::npos )
       partonMode_="Herwig++";
     else if( moduleName.find("Sherpa")!=std::string::npos )
       partonMode_="Sherpa";
     else
       partonMode_="Undefined";
   }

   // set the parton selection mode (done only once per job)
   if( !partonSelector_ )
   {
     if ( partonMode_=="Undefined" )
       edm::LogWarning("UndefinedPartonMode") << "Could not automatically determine the hadronizer type and set the correct parton selection mode. Parton-based jet flavour will not be defined.";
     else if ( partonMode_=="Pythia6" )
     {
       partonSelector_ = PartonSelectorPtr( new Pythia6PartonSelector() );
       edm::LogInfo("PartonModeDefined") << "Using Pythia6 parton selection mode.";
     }
     else if ( partonMode_=="Pythia8" )
     {
       partonSelector_ = PartonSelectorPtr( new Pythia8PartonSelector() );
       edm::LogInfo("PartonModeDefined") << "Using Pythia8 parton selection mode.";
     }
     else if ( partonMode_=="Herwig6" )
     {
       partonSelector_ = PartonSelectorPtr( new Herwig6PartonSelector() );
       edm::LogInfo("PartonModeDefined") << "Using Herwig6 parton selection mode.";
     }
     else if ( partonMode_=="Herwig++" )
     {
       partonSelector_ = PartonSelectorPtr( new HerwigppPartonSelector() );
       edm::LogInfo("PartonModeDefined") << "Using Herwig++ parton selection mode.";
     }
     else if ( partonMode_=="Sherpa" )
     {
       partonSelector_ = PartonSelectorPtr( new SherpaPartonSelector() );
       edm::LogInfo("PartonModeDefined") << "Using Sherpa parton selection mode.";
     }
     else
       throw cms::Exception("InvalidPartonMode") <<"Parton selection mode is invalid: " << partonMode_ << ", use Auto | Pythia6 | Pythia8 | Herwig6 | Herwig++ | Sherpa" << std::endl;
   }

   edm::Handle<reco::GenParticleCollection> particles;
   iEvent.getByToken(particlesToken_, particles);

   std::auto_ptr<reco::GenParticleRefVector> bHadrons ( new reco::GenParticleRefVector );
   std::auto_ptr<reco::GenParticleRefVector> cHadrons ( new reco::GenParticleRefVector );
   std::auto_ptr<reco::GenParticleRefVector> partons  ( new reco::GenParticleRefVector );
   std::auto_ptr<reco::GenParticleRefVector> leptons  ( new reco::GenParticleRefVector );

   // loop over particles and select b and c hadrons and leptons
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
       // check if any of the daughters is also a c hadron
       bool hascHadronDaughter = false;
       for(size_t i=0; i < it->numberOfDaughters(); ++i)
       {
         if( CandMCTagUtils::hasCharm( *(it->daughter(i)) ) ) { hascHadronDaughter = true; break; }
       }
       if( hascHadronDaughter ) continue; // skip excited c hadrons that have other c hadrons as daughters

       cHadrons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
     }

     // status==1 electrons and muons
     if( ( reco::isElectron( *it ) || reco::isElectron( *it ) ) && it->status()==1 )
       leptons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );

     // status==2 taus
     if( reco::isTau( *it ) && it->status()==2 )
       leptons->push_back( reco::GenParticleRef( particles, it - particles->begin() ) );
   }

   // select partons
   if ( partonMode_!="Undefined" )
     partonSelector_->run(particles,partons);

   iEvent.put( bHadrons, "bHadrons" );
   iEvent.put( cHadrons, "cHadrons" );
   iEvent.put( partons,  "partons" );
   iEvent.put( leptons,  "leptons" );
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
