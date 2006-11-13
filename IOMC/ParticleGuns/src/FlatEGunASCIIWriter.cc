// ----------------------------------------------------------------------
// FlatEGunASCIIWriter.cc
// Author: Julia Yarba
//
// This code has been molded after examples in HepMC and HepPDT, and
// after single particle gun example (private contacts with Lynn Garren)
//
// Plus, it uses the ParameterSet funtionalities for "user interface"  
//
// It creates a ParticleDataTable from PDG-2004 data file.
// It then creates one or several HepMC particle(s) and adds it(them) 
// to the HepMC::GenEvent.
// For particle definition it uses flat random gun in a given eta, phi 
// and energy ranges to calculate particle kinematics.
// Vertex smearing is currently off, can be easily added later.
// After all, it writes out the event into ASCII output file.
//
// ----------------------------------------------------------------------

#include <iostream>
// header
#include "IOMC/ParticleGuns/interface/FlatEGunASCIIWriter.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;
      
FlatEGunASCIIWriter::FlatEGunASCIIWriter( const ParameterSet& pset )
   : fEvt(0), 
     fPDGTable( new DefaultConfig::ParticleDataTable("PDG Table") ),
     fOutFileName( pset.getUntrackedParameter<string>("OutFileName","FlatEGunHepMC.dat") ),
     fOutStream( new ofstream( fOutFileName.c_str() ) ),
     fCurrentEvent(0)
{
      
  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  fPartIDs    = pgun_params.getParameter< vector<int> >("PartID");
  fMinEta     = pgun_params.getParameter<double>("MinEta");
  fMaxEta     = pgun_params.getParameter<double>("MaxEta");
  fMinPhi     = pgun_params.getParameter<double>("MinPhi");
  fMaxPhi     = pgun_params.getParameter<double>("MaxPhi");
  fMinE       = pgun_params.getParameter<double>("MinE");
  fMaxE       = pgun_params.getParameter<double>("MaxE");
  
    
  // 
  // fPDGTablePath = "/afs/cern.ch/sw/lcg/external/clhep/1.9.2.1/slc3_ia32_gcc323/data/HepPDT/" ;
  string HepPDTBase( getenv("HEPPDT_PARAM_PATH") ) ; 
  fPDGTablePath = HepPDTBase + "/data/" ;
  fPDGTableName = "PDG_mass_width_2004.mc";


  string TableFullName = fPDGTablePath + fPDGTableName ;
  ifstream PDFile( TableFullName.c_str() ) ;
  if( !PDFile ) 
  {
      throw cms::Exception("FileNotFound", "FlatEGunASCIIWriter::FlatEGunASCIIWriter()")
	<< "File " << TableFullName << " cannot be opened.\n";
  }

  HepPDT::TableBuilder tb(*fPDGTable) ;
  if ( !addPDGParticles( PDFile, tb ) ) { cout << " Error reading PDG !" << endl; }
  // the tb dtor fills fPDGTable

  cout << "HepMC Particle Gun ASCII Writer is initialized" << endl ;
  cout << "Requested # of Particles : " << fPartIDs.size() << endl ;
            
}
   
FlatEGunASCIIWriter::~FlatEGunASCIIWriter()
{
   
  if ( fEvt != NULL ) delete fEvt ;
  delete fPDGTable;
  delete fOutStream ;

}

void FlatEGunASCIIWriter::beginJob( const EventSetup& )
{

/*
  string TableFullName = fPDGTablePath + fPDGTableName ;
  ifstream PDFile( TableFullName.c_str() ) ;
  if( !PDFile ) 
  {
      throw cms::Exception("FileNotFound", "FlatEGunASCIIWriter::beginJob()")
	<< "File " << TableFullName << " cannot be opened.\n";
  }

  HepPDT::TableBuilder tb(*fPDGTable) ;
  if ( !addPDGParticles( PDFile, tb ) ) { cout << " Error reading PDG !" << endl; }
  // the tb dtor fills fPDGTable
*/
   fPDGTable->writeParticleData( *fOutStream ) ;
   HepMC::writeLegend( *fOutStream ) ;
   
   return ;

}

void FlatEGunASCIIWriter::analyze( const Event& , 
                                   const EventSetup& /* not used so far, thus is "empty" */ ) 
{
         
   // for testing purpose
   // cout << "Event # " << fCurrentEvent << endl ;
   
   // clean up GenEvent memory : also deletes all vtx/part in it
   // 
   if ( fEvt != NULL ) delete fEvt ;
   
   // here re-create fEvt (memory)
   fEvt = new HepMC::GenEvent() ;

   // now actualy, cook up the event from PDGTable and gun parameters
   //
   HepMC::GenVertex* Vtx = new HepMC::GenVertex( CLHEP::HepLorentzVector(0.,0.,0.) );

   // loop over particles
   //
   for ( unsigned int ip=0; ip<fPartIDs.size(); ip++ )
   {
       double energy = RandFlat::shoot( fMinE, fMaxE ) ;
       double eta    = RandFlat::shoot( fMinEta, fMaxEta ) ;
       double phi    = RandFlat::shoot( fMinPhi, fMaxPhi ) ;
       const DefaultConfig::ParticleData* 
          PData = fPDGTable->particle(HepPDT::ParticleID(abs(fPartIDs[ip]))) ;
       double mass   = PData->mass().value() ;
       double mom2   = energy*energy - mass*mass ;
       double mom    = 0. ;
       if ( mom2 > 0. ) mom = sqrt(mom2) ;
       double theta  = 2.*atan(exp(-eta)) ;
       double px     = mom*sin(theta)*cos(phi) ;
       double py     = mom*sin(theta)*sin(phi) ;
       double pz     = mom*cos(theta) ;
       CLHEP::Hep3Vector p(px,py,pz) ;
       HepMC::GenParticle* Part = 
           new HepMC::GenParticle(CLHEP::HepLorentzVector(p,energy),fPartIDs[ip],1);
       Vtx->add_particle_out(Part);
   }
   fEvt->add_vertex( Vtx ) ;
   fEvt->set_event_number( fCurrentEvent+1 ) ;
   fEvt->set_signal_process_id(20) ; 
   
   // for testing purpose only
   // fEvt->print() ;     

   (*fOutStream) << fEvt ;
   
   fCurrentEvent++ ;

   return ;
   
}

