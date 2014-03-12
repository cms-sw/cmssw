/* this is the code for the new Tauola++ */

#include <iostream>

#include "GeneratorInterface/TauolaInterface/interface/TauolappInterface.h"

#include "Tauola.h"
#include "TauolaHepMCEvent.h"
#include "Log.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandomEngine.h"

#include "HepMC/GenEvent.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

extern "C" {

  void gen::ranmar_( float *rvec, int *lenv )
  {
      TauolappInterface* instance = TauolappInterface::getInstance();
      for(int i = 0; i < *lenv; i++)
	 *rvec++ = instance->flat();
      return;
  }
  
  void gen::rmarin_( int*, int*, int* )
  {
     return;
  }

}

using namespace gen;
using namespace edm;
using namespace std;

TauolappInterface* TauolappInterface::fInstance = 0;


TauolappInterface::TauolappInterface(){
  Setup();
}

TauolappInterface::TauolappInterface( const edm::ParameterSet& pset){
  Setup();
  setPSet(pset);
}

void TauolappInterface::Setup(){
  fInstance=this;
  fPolarization=false;
  fPSet=0;
  fIsInitialized=false; fMDTAU=-1; 
  fSelectDecayByEvent=false;

  Service<RandomNumberGenerator> rng;
  if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
      << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
          "which appears to be absent.  Please add that service to your configuration\n"
      "or remove the modules that require it." << std::endl;
  }

  fRandomEngine = &rng->getEngine();

}


//TauolappInterface::TauolappInterface( const ParameterSet& pset )
//   : fIsInitialized(false)
//{
//
//   Tauola::setDecayingParticle(15);
//   // --> ??? Tauola::setRadiation(false);
//
//   // polarization switch 
//   //
//   // fPolarization = pset.getParameter<bool>("UseTauolaPolarization") ? 1 : 0 ;
//   fPolarization = pset.getParameter<bool>("UseTauolaPolarization");
//   
//   // read tau decay mode switches
//   //
//   ParameterSet cards = pset.getParameter< ParameterSet >("InputCards");
//   Tauola::setSameParticleDecayMode( cards.getParameter< int >( "pjak1" ) ) ;
//   Tauola::setOppositeParticleDecayMode( cards.getParameter< int >( "pjak2" ) ) ;
//
//   Tauola::setTauLifetime(0.0);
//   Tauola::spin_correlation.setAll(fPolarization);
//
//   // some more options, copied over from an example 
//   // - maybe will use later...
//   //
//   //Tauola::setEtaK0sPi(0,0,0); // switches to decay eta K0_S and pi0 1/0 on/off. 
//   //
//
//} 


TauolappInterface* TauolappInterface::getInstance()
{

   if ( fInstance == 0 ) fInstance = new TauolappInterface() ;
   return fInstance;

}


TauolappInterface::~TauolappInterface()
{

   if ( fPSet != 0 ) delete fPSet;
   if ( fInstance == this ) fInstance = 0;

}

void TauolappInterface::setPSet( const ParameterSet& pset )
{

   if ( fPSet != 0 ) 
   {
      throw cms::Exception("TauolappInterfaceError")
         << "Attempt to override Tauola an existing ParameterSet\n"
         << std::endl;   
   }
   
   fPSet = new ParameterSet(pset);
   
   return;

}

void TauolappInterface::init( const edm::EventSetup& es )
{
   
   if ( fIsInitialized ) return; // do init only once
   
   if ( fPSet == 0 ) 
   {

      throw cms::Exception("TauolappInterfaceError")
         << "Attempt to initialize Tauola with an empty ParameterSet\n"
         << std::endl;   
   }
   
   fIsInitialized = true;
      
   es.getData( fPDGTable ) ;

   Tauola::setDecayingParticle(15);
   // --> ??? Tauola::setRadiation(false);

   // polarization switch 
   //
   // fPolarization = fPSet->getParameter<bool>("UseTauolaPolarization") ? 1 : 0 ;
   fPolarization = fPSet->getParameter<bool>("UseTauolaPolarization");
   
   // read tau decay mode switches
   //
   ParameterSet cards = fPSet->getParameter< ParameterSet >("InputCards");
   
   fMDTAU = cards.getParameter< int >( "mdtau" );

   if ( fMDTAU == 0 || fMDTAU == 1 )
   {
      Tauola::setSameParticleDecayMode( cards.getParameter< int >( "pjak1" ) ) ;
      Tauola::setOppositeParticleDecayMode( cards.getParameter< int >( "pjak2" ) ) ;
   }

   Tauola::setTauLifetime(0.0);
   Tauola::spin_correlation.setAll(fPolarization);

   // some more options, copied over from an example 
   // - maybe will use later...
   //
   //Tauola::setEtaK0sPi(0,0,0); // switches to decay eta K0_S and pi0 1/0 on/off. 
   //

//
//   const HepPDT::ParticleData* 
//         PData = fPDGTable->particle(HepPDT::ParticleID( abs(Tauola::getDecayingParticle()) )) ;
//   double lifetime = PData->lifetime().value();
//   Tauola::setTauLifetime( lifetime );

   fPDGs.push_back( Tauola::getDecayingParticle() );

   Tauola::setRandomGenerator(&gen::TauolappInterface_RandGetter);         
   Tauola::initialize();

   Tauola::spin_correlation.setAll(fPolarization);// Tauola switches this on during Tauola::initialise(); so we add this here to keep it on/off

   // override decay modes if needs be
   //
   // we have to do it AFTER init because otherwises branching ratios are NOT filled in
   //
   if ( fMDTAU != 0 && fMDTAU != 1 )
   {
      decodeMDTAU( fMDTAU );
   }

   Log::LogWarning(false);
   
   return;
}

float TauolappInterface::flat()
{

   if ( !fPSet )
   {
      // throw
      throw cms::Exception("TauolappInterfaceError")
         << "Attempt to run random number generator of un-initialized Tauola\n"
         << std::endl;   
   }
   
   if ( !fIsInitialized ) 
   {
      // throw
      throw cms::Exception("TauolappInterfaceError")
         << "Attempt to run random number generator of un-initialized Tauola\n"
         << std::endl;   
   }
   
   return fRandomEngine->flat();

}

HepMC::GenEvent* TauolappInterface::decay( HepMC::GenEvent* evt )
{
      
   if ( !fIsInitialized ) return evt;
   
   int NPartBefore = evt->particles_size();
   int NVtxBefore  = evt->vertices_size();
   
   // what do we do if Hep::GenEvent size is larger than 10K ???
   // Tauola (& Photos, BTW) can only handle up to 10K via HEPEVT,
   // and in case of CMS, it's only up to 4K !!!
   //
   // if ( NPartBefore > 10000 ) return evt;
   //
   
   // override decay mode if needs be
   if ( fSelectDecayByEvent )
   {
      selectDecayByMDTAU();
   }
   
    //construct tmp TAUOLA event
    //
    auto * t_event = new TauolaHepMCEvent(evt);
   
    // another option: if one lets Pythia or another master gen to decay taus, 
    //                 we have to undecay them first
    // t_event->undecayTaus();
    
    // run Tauola on the tmp event - HepMC::GenEvernt will be MODIFIED !!!
    //
    t_event->decayTaus();
    
    // delet tmp Tauola event
    //
    delete t_event; 
    
    // do we also need to apply the lifetime and vtx position shift ??? 
    // (see TauolappInterface, for example)
    //
    // NOTE: the procedure ASSYMES that vertex barcoding is COUNTIUOUS/SEQUENTIAL,
    // and that the abs(barcode) corresponds to vertex "plain indexing"
    //
    for ( int iv=NVtxBefore+1; iv<=evt->vertices_size(); iv++ )
    {
       HepMC::GenVertex* GenVtx = evt->barcode_to_vertex(-iv);
       HepMC::GenParticle* GenPart = *(GenVtx->particles_in_const_begin());
       HepMC::GenVertex* ProdVtx = GenPart->production_vertex();
       HepMC::FourVector PMom = GenPart->momentum();
       double mass = GenPart->generated_mass();
       const HepPDT::ParticleData* 
             PData = fPDGTable->particle(HepPDT::ParticleID(abs(GenPart->pdg_id()))) ;
       double lifetime = PData->lifetime().value();
       float prob = 0.;
       int length=1;
       ranmar_(&prob,&length);
       double ct = -lifetime * std::log(prob);
       double VxDec = GenVtx->position().x();
       VxDec += ct * (PMom.px()/mass);
       VxDec += ProdVtx->position().x();
       double VyDec = GenVtx->position().y();
       VyDec += ct * (PMom.py()/mass);
       VyDec += ProdVtx->position().y();
       double VzDec = GenVtx->position().z();
       VzDec += ct * (PMom.pz()/mass);
       VzDec += ProdVtx->position().z();
       double VtDec = GenVtx->position().t();
       VtDec += ct * (PMom.e()/mass);
       VtDec += ProdVtx->position().t();
       GenVtx->set_position( HepMC::FourVector(VxDec,VyDec,VzDec,VtDec) ); 
       //
       // now find decay products with funky barcode, weed out and replace with clones of sensible barcode
       // we can NOT change the barcode while iterating, because iterators do depend on the barcoding
       // thus we have to take a 2-step procedure
       //
       std::vector<int> BCodes;
       BCodes.clear();
       for (HepMC::GenVertex::particle_iterator pitr= GenVtx->particles_begin(HepMC::children);
                                               pitr != GenVtx->particles_end(HepMC::children); ++pitr) 
       {
	  if ( (*pitr)->barcode() > 10000 )
	  {
	     BCodes.push_back( (*pitr)->barcode() );
	  }
       }
       if ( BCodes.size() > 0 )
       {
          for ( size_t ibc=0; ibc<BCodes.size(); ibc++ )
	  {
	     HepMC::GenParticle* p1 = evt->barcode_to_particle( BCodes[ibc] );
	     int nbc = p1->barcode() - 10000 + NPartBefore;
             p1->suggest_barcode( nbc );
	  }
       }             
    }
        
    return evt;
      
}

void TauolappInterface::statistics()
{
   return;
}

void TauolappInterface::decodeMDTAU( int mdtau )
{

   // Note-1:
   // I have to hack the common block directly because set<...>DecayMode(...)
   // only changes it in the Tauola++ instance but does NOT passes it over
   // to the Fortran core - this it does only one, via initialize() stuff...
   //
   // So I'll do both ways of settings, just for consistency...
   // but I probably need to communicate it to the Tauola(++) team...
   //
   
   // Note-2: 
   // originally, the 1xx settings are meant for tau's from hard event,
   // and the 2xx settings are for any tau in the event record;
   //
   // later one, we'll have to take this into account...
   // but first I'll have to sort out what happens in the 1xx case
   // to tau's coming outside of hard event (if any in the record)   
   //
   
   if ( mdtau == 101 || mdtau == 201 )
   {
      // override with electron mode for both tau's
      //
      jaki_.jak1 = 1;
      jaki_.jak2 = 1;
      Tauola::setSameParticleDecayMode( 1 ) ;
      Tauola::setOppositeParticleDecayMode( 1 ) ;
      return;
   }
   
   if ( mdtau == 102 || mdtau == 202 )
   {
      // override with muon mode for both tau's
      //
      jaki_.jak1 = 2;
      jaki_.jak2 = 2;
      Tauola::setSameParticleDecayMode( 2 ) ;
      Tauola::setOppositeParticleDecayMode( 2 ) ;
      return;
   }

   if ( mdtau == 111 || mdtau == 211 )
   {
      // override with electron mode for 1st tau 
      // and any mode for 2nd tau
      //
      jaki_.jak1 = 1;
      jaki_.jak2 = 0;
      Tauola::setSameParticleDecayMode( 1 ) ;
      Tauola::setOppositeParticleDecayMode( 0 ) ;
      return;
   }

   if ( mdtau == 112 || mdtau == 212 )
   {
      // override with muon mode for the 1st tau 
      // and any mode for the 2nd tau
      //
      jaki_.jak1 = 2;
      jaki_.jak2 = 0;
      Tauola::setSameParticleDecayMode( 2 ) ;
      Tauola::setOppositeParticleDecayMode( 0 ) ;
      return;
   }
   
   if ( mdtau == 121 || mdtau == 221 )
   {
      // override with any mode for the 1st tau 
      // and electron mode for the 2nd tau
      //
      jaki_.jak1 = 0;
      jaki_.jak2 = 1;
      Tauola::setSameParticleDecayMode( 0 ) ;
      Tauola::setOppositeParticleDecayMode( 1 ) ;
      return;
   }
   
   if ( mdtau == 122 || mdtau == 222 )
   {
      // override with any mode for the 1st tau 
      // and muon mode for the 2nd tau
      //
      jaki_.jak1 = 0;
      jaki_.jak2 = 2;
      Tauola::setSameParticleDecayMode( 0 ) ;
      Tauola::setOppositeParticleDecayMode( 2 ) ;
      return;
   }

   if ( mdtau == 140 || mdtau == 240 )
   {
      // override with pi+/- nutau mode for both tau's 
      //
      jaki_.jak1 = 3;
      jaki_.jak2 = 3;
      Tauola::setSameParticleDecayMode( 3 ) ;
      Tauola::setOppositeParticleDecayMode( 3 ) ;
      return;
   }

   if ( mdtau == 141 || mdtau == 241 )
   {
      // override with pi+/- nutau mode for the 1st tau 
      // and any mode for the 2nd tau
      //
      jaki_.jak1 = 3;
      jaki_.jak2 = 0;
      Tauola::setSameParticleDecayMode( 3 ) ;
      Tauola::setOppositeParticleDecayMode( 0 ) ;
      return;
   }

   if ( mdtau == 142 || mdtau == 242 )
   {
      // override with any mode for the 1st tau 
      // and pi+/- nutau mode for 2nd tau
      //
      jaki_.jak1 = 0;
      jaki_.jak2 = 3;
      Tauola::setSameParticleDecayMode( 0 ) ;
      Tauola::setOppositeParticleDecayMode( 3 ) ;
      return;
   }
   
   // OK, we come here for semi-inclusive modes
   //
   
   // First of all, leptons and hadron modes sums
   //
   // re-scale branching ratios, just in case...
   //
   double sumBra = 0;
   
   // the number of decay modes is hardcoded at 22 because that's what it is right now in Tauola
   // in the future, perhaps an asscess method would be useful - communicate to Tauola team...
   //
   
   for ( int i=0; i<22; i++ )
   {
      sumBra += taubra_.gamprt[i];
   }
   if ( sumBra == 0. ) return ; // perhaps need to throw ?
   for ( int i=0; i<22; i++ )
   {
      double newBra = taubra_.gamprt[i] / sumBra;
      Tauola::setTauBr( i+1, newBra ); 
   }
   sumBra = 1.0;
   
   double sumLeptonBra = taubra_.gamprt[0] + taubra_.gamprt[1];
   double sumHadronBra = sumBra - sumLeptonBra;
   
   for ( int i=0; i<2; i++ )
   {
      fLeptonModes.push_back( i+1 );
      fScaledLeptonBrRatios.push_back( (taubra_.gamprt[i]/sumLeptonBra) );  
   }
   for ( int i=2; i<22; i++ )
   {
      fHadronModes.push_back( i+1 );
      fScaledHadronBrRatios.push_back( (taubra_.gamprt[i]/sumHadronBra) ); 
   }

   fSelectDecayByEvent = true;
   return;
      
}

void TauolappInterface::selectDecayByMDTAU()
{

   
   if ( fMDTAU == 100 || fMDTAU == 200 )
   {
      int mode = selectLeptonic();
      jaki_.jak1 = mode;
      Tauola::setSameParticleDecayMode( mode );
      mode = selectLeptonic();
      jaki_.jak2 = mode;
      Tauola::setOppositeParticleDecayMode( mode );
      return ;
   }
   
   int modeL = selectLeptonic();
   int modeH = selectHadronic();
   
   if ( fMDTAU == 110 || fMDTAU == 210 )
   {
      jaki_.jak1 = modeL;
      jaki_.jak2 = 0;
      Tauola::setSameParticleDecayMode( modeL );
      Tauola::setOppositeParticleDecayMode( 0 );
      return ;
   }
   
   if ( fMDTAU == 120 || fMDTAU == 22 )
   {
      jaki_.jak1 = 0;
      jaki_.jak2 = modeL;
      Tauola::setSameParticleDecayMode( 0 );
      Tauola::setOppositeParticleDecayMode( modeL );
      return;      
   }
   
   if ( fMDTAU == 114 || fMDTAU == 214 )
   {
      jaki_.jak1 = modeL;
      jaki_.jak2 = modeH;
      Tauola::setSameParticleDecayMode( modeL );
      Tauola::setOppositeParticleDecayMode( modeH );
      return;      
   }

   if ( fMDTAU == 124 || fMDTAU == 224 )
   {
      jaki_.jak1 = modeH;
      jaki_.jak2 = modeL;
      Tauola::setSameParticleDecayMode( modeH );
      Tauola::setOppositeParticleDecayMode( modeL );
      return;      
   }

   if ( fMDTAU == 115 || fMDTAU == 215 )
   {
      jaki_.jak1 = 1;
      jaki_.jak2 = modeH;
      Tauola::setSameParticleDecayMode( 1 );
      Tauola::setOppositeParticleDecayMode( modeH );
      return;      
   }

   if ( fMDTAU == 125 || fMDTAU == 225 )
   {
      jaki_.jak1 = modeH;
      jaki_.jak2 = 1;
      Tauola::setSameParticleDecayMode( modeH );
      Tauola::setOppositeParticleDecayMode( 1 );
      return;      
   }

   if ( fMDTAU == 116 || fMDTAU == 216 )
   {
      jaki_.jak1 = 2;
      jaki_.jak2 = modeH;
      Tauola::setSameParticleDecayMode( 2 );
      Tauola::setOppositeParticleDecayMode( modeH );
      return;      
   }

   if ( fMDTAU == 126 || fMDTAU == 226 )
   {
      jaki_.jak1 = modeH;
      jaki_.jak2 = 2;
      Tauola::setSameParticleDecayMode( modeH );
      Tauola::setOppositeParticleDecayMode( 2 );
      return;      
   }

   if ( fMDTAU == 130 || fMDTAU == 230 )
   {
      jaki_.jak1 = modeH;
      jaki_.jak2 = selectHadronic();
      Tauola::setSameParticleDecayMode( modeH );
      Tauola::setOppositeParticleDecayMode( jaki_.jak2 );
      return;      
   }

   if ( fMDTAU == 131 || fMDTAU == 231 )
   {
      jaki_.jak1 = modeH;
      jaki_.jak2 = 0;
      Tauola::setSameParticleDecayMode( modeH );
      Tauola::setOppositeParticleDecayMode( 0 );
      return;      
   }

   if ( fMDTAU == 132 || fMDTAU == 232 )
   {
      jaki_.jak1 = 0;
      jaki_.jak2 = modeH;
      Tauola::setSameParticleDecayMode( 0 );
      Tauola::setOppositeParticleDecayMode( modeH );
      return;      
   }
   
   // unlikely that we get here on unknown mdtau 
   // - there's a protection earlier
   // but if we do, just set defaults
   // probably need to spit a warning...
   //
   Tauola::setSameParticleDecayMode( 0 );
   Tauola::setOppositeParticleDecayMode( 0 );
      
   return;
   

}

int TauolappInterface::selectLeptonic()
{
   
   float prob = flat();
   
   if ( prob > 0. && prob <= fScaledLeptonBrRatios[0] ) 
   {
      return 1;
   }
   else if ( prob > fScaledLeptonBrRatios[1] && prob <=1. )
   {
      return 2;
   }
      
   return 0;
}

int TauolappInterface::selectHadronic()
{

   float prob = 0.;
   int len = 1;
   ranmar_(&prob,&len);
   
   double sumBra = fScaledHadronBrRatios[0];
   if ( prob > 0. && prob <= sumBra ) 
   {
      return fHadronModes[0];
   }
   else
   {
      int NN = fScaledHadronBrRatios.size();
      for ( int i=1; i<NN; i++ )
      {
         if ( prob > sumBra && prob <= (sumBra+fScaledHadronBrRatios[i]) ) 
	 {
	    return fHadronModes[i];
	 }
	 sumBra += fScaledHadronBrRatios[i];
      }
   }
   
   return 0;

}

double gen::TauolappInterface_RandGetter(){
  TauolappInterface* instance = TauolappInterface::getInstance();
  return  (double)instance->flat();
}

