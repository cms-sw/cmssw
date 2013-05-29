/* for old tauola27 */
#include <iostream>

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

#include "GeneratorInterface/ExternalDecays/interface/TauolaInterface.h"
#include "GeneratorInterface/ExternalDecays/interface/TauolaWrapper.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "HepMC/GenEvent.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "GeneratorInterface/ExternalDecays/interface/DecayRandomEngine.h"

using namespace gen;
using namespace edm;
using namespace std;

extern "C" {

  void ranmar_( float *rvec, int *lenv )
  {

      for(int i = 0; i < *lenv; i++)
         *rvec++ = decayRandomEngine->flat();

      return;

  }
  
  void rmarin_( int*, int*, int* )
  {

     return;

  }

}

//
//   General Note: While there're no explicit calls or otherwise "links" to Pythia6 anywhere,
//   we're using Pythia6Service here because we run pretauola rather than "core" tauola;
//   pretauola is an extension on top of tauola, which is tied to Pythia6 via several routines;
//   most heavily use one is PYR - we can't avoid it (other Pythia6-tied routines we avoid)
//

TauolaInterface::TauolaInterface( const ParameterSet& pset )
   : fIsInitialized(false)
{
   fPy6Service = new Pythia6Service;

   fPolarization = pset.getParameter<bool>("UseTauolaPolarization") ? 1 : 0 ;

   // set Tauola defaults
   //
   ki_taumod_.pjak1 = -1;
   ki_taumod_.pjak2 = -1;
   ki_taumod_.mdtau = -1;
   
   // read tau decay mode switches
   //
   ParameterSet cards = pset.getParameter< ParameterSet >("InputCards");
   ki_taumod_.pjak1 = cards.getParameter< int >( "pjak1" );
   ki_taumod_.pjak2 = cards.getParameter< int >( "pjak2" );
   ki_taumod_.mdtau = cards.getParameter< int >( "mdtau" );
   
} 

TauolaInterface::~TauolaInterface()
{
   delete fPy6Service;
}

void TauolaInterface::init( const edm::EventSetup& es )
{
   
   if ( fIsInitialized ) return; // do init only once
   
   if ( ki_taumod_.mdtau <= -1 ) // actually, need to throw exception !
      return ;
   
   fPDGs.push_back( 15 ) ;
   es.getData( fPDGTable ) ;

	cout << "----------------------------------------------" << endl;
        cout << "Initializing Tauola" << endl;
	if ( fPolarization == 0 )
	{
	   cout << "Tauola: Polarization disabled" << endl;
	} 
	else if ( fPolarization == 1 )
	{
	   cout << "Tauola: Polarization enabled" << endl;
	}

// FIXME !!!
// This is a temporary hack - we're re-using master generator's seed to init RANMAR
// FIXME !!!
//   This is now off because ranmar has been overriden (see code above) to use pyr_(...)
//   - this way we're using guaranteed initialized rndm generator... BUT !!! in the long
//   run we may want a separate random stream for tauola...

//   Service<RandomNumberGenerator> rng;
//   int seed = rng->mySeed() ;
//   int ntot=0, ntot2=0;
//   rmarin_( &seed, &ntot, &ntot2 );

   int mode = -2;
   taurep_( &mode ) ;
   mode = -1;
   // tauola_( &mode, &fPolarization );
   // tauola_srs_( &mode, &fPolarization );
   //
   // We're using the call(...) method here because it'll make sure that Py6 
   // is initialized, and that's done only once, and will grab exatly that instance
   //
   fPy6Service->call( tauola_srs_, &mode, &fPolarization ); 
   
   fIsInitialized = true;
   
   return;
}

HepMC::GenEvent* TauolaInterface::decay( HepMC::GenEvent* evt )
{
   
   // event record convertor
   //
   HepMC::IO_HEPEVT conv;
   
   if ( !fIsInitialized ) return conv.read_next_event();
   
   // We are using random numbers, we are fetched through Pythia6Service
   // (through ranmar_ below) -> so grab the instance during decay()

   Pythia6Service::InstanceWrapper pythia6InstanceGuard( fPy6Service );

   // fill up HEPEVT common block
   //
   // IDEALLY, this should be the way to go
   // BUT !!! this utility fills it up in the "reshuffled" order,
   // and later on Tauola chocks on it 
   //
   // Needs to be sorted out, eith in HepMC, or in Tauola, or both !!!
   // 
   // At present, this thing blindly relies on the assumption that
   // HEPEVT is always there - which wont be the case with Py8 or Hwg++
   //
   //HepMC::IO_HEPEVT conv;
   //conv.write_event( evt ) ;
   
   int numPartBeforeTauola = HepMC::HEPEVT_Wrapper::number_entries();
   // HepMC::HEPEVT_Wrapper::print_hepevt();
   
   int mode = 0;
   // tauola_( &mode, &fPolarization );
   fPy6Service->call( tauola_srs_, &mode, &fPolarization );
   
   int numPartAfterTauola = HepMC::HEPEVT_Wrapper::number_entries();
   // HepMC::HEPEVT_Wrapper::print_hepevt();
   
   // before we do the conversion, we need to deal with decay vertexes
   // since Tauola knows nothing about lifetimes, all decay vertexes are set to 0. 
   // nees to set them properly, knowing lifetime !
   // here we do it on HEPEVT record, also for consistency, although it's probably
   // even easier to deal with HepMC::GenEvent record  
   
   // find 1st "non-doc" tau
   //
   bool foundTau = false;
   for ( int ip=1; ip<=numPartAfterTauola; ip++ )
   {
      if ( std::abs( HepMC::HEPEVT_Wrapper::id( ip ) ) == 15
           && HepMC::HEPEVT_Wrapper::status( ip ) != 3 )
      {
         foundTau = true;
	 break;
      }
   }
   
   if ( !foundTau )
   {
      // no tau found
      // just give up here
      //
      return conv.read_next_event();
   }
   
   std::vector<int> PrntIndx;
   
   for ( int ip=numPartAfterTauola; ip>numPartBeforeTauola; ip-- ) // Fortran indexing !
   {
      
      // first of all, find out how many generations in decay chain
      //
      PrntIndx.clear();
      int Prnt = HepMC::HEPEVT_Wrapper::first_parent(ip);
      ip -= (HepMC::HEPEVT_Wrapper::number_children(Prnt)-1); // such that we don't go the same part again
      PrntIndx.push_back( Prnt );
      while ( abs( HepMC::HEPEVT_Wrapper::id(Prnt) ) != 15 ) // shortcut; need to loop over fPDGs...
      {
	 int Prnt1 = HepMC::HEPEVT_Wrapper::first_parent( Prnt );
	 Prnt = Prnt1;
	 // such that the tau always appear at the start of the list
	 PrntIndx.insert( PrntIndx.begin(), Prnt );
         ip -= HepMC::HEPEVT_Wrapper::number_children(Prnt); // such that we don't go the same part again
      }
      for ( size_t iprt=0; iprt<PrntIndx.size(); iprt++ )
      {  
          int Indx = PrntIndx[iprt];
	  int PartID = HepMC::HEPEVT_Wrapper::id( Indx );
	  const HepPDT::ParticleData* 
             PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID))) ;
	 //
	 // prob = exp(-t/lifetime) ==> t = -lifetime * log(prob)
	 //
	 float prob = 0.;
	 int length=1;
	 ranmar_(&prob,&length);
	 double lifetime = PData->lifetime().value();
	 //
	 // in case of Py6, this would be copied into V(5,i)
	 // for HEPEVT, need to check...
	 //
	 double ct = -lifetime * std::log(prob);
	 //
	 double ee = HepMC::HEPEVT_Wrapper::e( Indx );
	 double px = HepMC::HEPEVT_Wrapper::px( Indx );
	 double py = HepMC::HEPEVT_Wrapper::py( Indx );
	 double pz = HepMC::HEPEVT_Wrapper::pz( Indx );
	 // double pp = std::sqrt( px*px + py*py + pz*pz );
	 double mass = HepMC::HEPEVT_Wrapper::m( Indx );
	 //
	 // this is in py6 terms:
	 //  VDCY(J)=V(IP,J)+V(IP,5)*P(IP,J)/P(IP,5)
	 //
	 double VxDec = HepMC::HEPEVT_Wrapper::x( Indx );
	 VxDec += ct * (px/mass);
	 double VyDec = HepMC::HEPEVT_Wrapper::y( Indx );
	 VyDec += ct * (py/mass);
	 double VzDec = HepMC::HEPEVT_Wrapper::z( Indx );
	 VzDec += ct * (pz/mass);
	 double VtDec = HepMC::HEPEVT_Wrapper::t( Indx );
	 VtDec += ct * (ee/mass);
	 for ( int idau=HepMC::HEPEVT_Wrapper::first_child( Indx );
	           idau<=HepMC::HEPEVT_Wrapper::last_child( Indx ); idau++ )
	 {
	    HepMC::HEPEVT_Wrapper::set_position( idau, VxDec, VyDec, VzDec, VtDec );
	 }
      }
   }
   
   return conv.read_next_event();
      
}

void TauolaInterface::statistics()
{
   int mode = 1;
   // tauola_( &mode, &fPolarization ) ;
   // tauola_srs_( &mode, &fPolarization ) ;
   fPy6Service->call( tauola_srs_, &mode, &fPolarization );
   return;
}

/* */

/* this is the code for the new Tauola++ 

#include <iostream>

#include "GeneratorInterface/ExternalDecays/interface/TauolaInterface.h"

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

// #include "GeneratorInterface/ExternalDecays/interface/DecayRandomEngine.h"


extern "C" {

  void gen::ranmar_( float *rvec, int *lenv )
  {
      TauolaInterface* instance = TauolaInterface::getInstance();
      for(int i = 0; i < *lenv; i++)
         // *rvec++ = decayRandomEngine->flat();
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

TauolaInterface* TauolaInterface::fInstance = 0;


TauolaInterface::TauolaInterface()
   : fPolarization(false), fPSet(0), fIsInitialized(false)
{
   
   Service<RandomNumberGenerator> rng;
   if(!rng.isAvailable()) {
    throw cms::Exception("Configuration")
       << "The RandomNumberProducer module requires the RandomNumberGeneratorService\n"
          "which appears to be absent.  Please add that service to your configuration\n"
          "or remove the modules that require it." << std::endl;
   }
   
   fRandomEngine = &rng->getEngine();

}


//TauolaInterface::TauolaInterface( const ParameterSet& pset )
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


TauolaInterface* TauolaInterface::getInstance()
{

   if ( fInstance == 0 ) fInstance = new TauolaInterface() ;
   return fInstance;

}


TauolaInterface::~TauolaInterface()
{

   if ( fPSet != 0 ) delete fPSet;
   if ( fInstance == this ) fInstance = 0;

}

void TauolaInterface::setPSet( const ParameterSet& pset )
{

   if ( fPSet != 0 ) 
   {
      throw cms::Exception("TauolaInterfaceError")
         << "Attempt to override Tauola an existing ParameterSet\n"
         << std::endl;   
   }
   
   fPSet = new ParameterSet(pset);
   
   return;

}

void TauolaInterface::init( const edm::EventSetup& es )
{
   
   if ( fIsInitialized ) return; // do init only once
   
   if ( fPSet == 0 ) 
   {

      throw cms::Exception("TauolaInterfaceError")
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
   Tauola::setSameParticleDecayMode( cards.getParameter< int >( "pjak1" ) ) ;
   Tauola::setOppositeParticleDecayMode( cards.getParameter< int >( "pjak2" ) ) ;

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
         
   Tauola::initialise();
   Log::LogWarning(false);
   
   return;
}

float TauolaInterface::flat()
{

   if ( !fPSet )
   {
      // throw
      throw cms::Exception("TauolaInterfaceError")
         << "Attempt to run random number generator of un-initialized Tauola\n"
         << std::endl;   
   }
   
   if ( !fIsInitialized ) 
   {
      // throw
      throw cms::Exception("TauolaInterfaceError")
         << "Attempt to run random number generator of un-initialized Tauola\n"
         << std::endl;   
   }
   
   return fRandomEngine->flat();

}

HepMC::GenEvent* TauolaInterface::decay( HepMC::GenEvent* evt )
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
   
    //construct tmp TAUOLA event
    //
    TauolaHepMCEvent * t_event = new TauolaHepMCEvent(evt);
   
    // another option: if one lets Pythia or another master gen to decay taus, 
    //                 we have to undecay them first
    // t_event->undecayTaus();
    
    // run Tauola on the tmp event - HepMC::GenEvernt will be MODIFIED !!!
    //
    t_event->decayTaus();
    
    // delet tmp Tauola event
    //
    delete t_event; 
    
    // fix barcodes of the newly added particles
    //
    for(HepMC::GenEvent::particle_const_iterator it = evt->particles_begin(); 
        it != evt->particles_end(); ++it ) 
    {
       //HepMC::GenParticle* GenPrt = (*it);
       if ( (*it)->barcode() > 10000 )
       {
          int NewBarcode = ((*it)->barcode()-10000) + NPartBefore;
	  (*it)->suggest_barcode( NewBarcode ); 
       }
    }
    

    // do we also need to apply the lifetime and vtx position shift ??? 
    // (see TauolaInterface, for example)
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
    }
    
    return evt;
      
}

void TauolaInterface::statistics()
{
   return;
}

*/
