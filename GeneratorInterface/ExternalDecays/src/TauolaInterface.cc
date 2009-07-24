
#include <iostream>

#include "GeneratorInterface/Pythia6Interface/interface/Pythia6Service.h"

#include "GeneratorInterface/ExternalDecays/interface/TauolaInterface.h"
#include "GeneratorInterface/ExternalDecays/interface/TauolaWrapper.h"

#include "HepMC/GenEvent.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

using namespace gen;
using namespace edm;
using namespace std;

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
/*
   Service<RandomNumberGenerator> rng;
   long seed = (long)(rng->mySeed()) ;
   fRandomEngine = new CLHEP::HepJamesRandom(seed) ;
   fRandomGenerator = new CLHEP::RandFlat(fRandomEngine) ;
*/   
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
   int mode = -2;
   taurep_( &mode ) ;
   mode = -1;
   // tauola_( &mode, &fPolarization );
   tauola_srs_( &mode, &fPolarization );
   
   fIsInitialized = true;
   
   return;
}

HepMC::GenEvent* TauolaInterface::decay( const HepMC::GenEvent* evt )
{
   
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
   
   HepMC::IO_HEPEVT conv;

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
      for ( int iprt=0; iprt<PrntIndx.size(); iprt++ )
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
	 double pp = std::sqrt( px*px + py*py + pz*pz );
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
   tauola_srs_( &mode, &fPolarization ) ;
}
