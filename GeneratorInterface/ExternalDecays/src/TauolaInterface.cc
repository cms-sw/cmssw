
#include <iostream>

#include "GeneratorInterface/ExternalDecays/interface/TauolaInterface.h"

#include "HepMC/GenEvent.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"


extern "C" {
  void tauola_(int*, int*);
}
extern "C" {
  extern struct {
    int pjak1;
    int pjak2;
    int mdtau;
  } ki_taumod_;
}
#define ki_taumod ki_taumod_

extern "C" {
   extern struct {
     int jak1;
     int jak2;
     int itdkrc;
     int ifphot;
     int ifhadm;
     int ifhadp;
   } libra_ ;
}
#define libra libra_

using namespace gen;
using namespace edm;
using namespace std;

TauolaInterface::TauolaInterface( const ParameterSet& pset )
{

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
}

void TauolaInterface::init()
{
   
   if ( ki_taumod_.mdtau <= -1 ) // actually, need to throw exception !
      return ;
   
   fPDGs.push_back( 15 ) ;
   
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
   int mode = -1;
   tauola_( &mode, &fPolarization );
   
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
   tauola_( &mode, &fPolarization );
   
   int numPartAfterTauola = HepMC::HEPEVT_Wrapper::number_entries();
   // HepMC::HEPEVT_Wrapper::print_hepevt();
   
   HepMC::IO_HEPEVT conv;
   return conv.read_next_event();
      
}

void TauolaInterface::statistics()
{
   int mode = 1;
   tauola_( &mode, &fPolarization ) ;
}
