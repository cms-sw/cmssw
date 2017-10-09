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
#include <fstream>
// header
#include "IOMC/ParticleGuns/interface/FlatEGunASCIIWriter.h"

// essentials !!!
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;
      
FlatEGunASCIIWriter::FlatEGunASCIIWriter( const ParameterSet& pset )
   : fEvt(0), 
     fOutFileName( pset.getUntrackedParameter<string>("OutFileName","FlatEGunHepMC.dat") ),
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
            
}
   
FlatEGunASCIIWriter::~FlatEGunASCIIWriter()
{
   
  if ( fEvt != NULL ) delete fEvt ;
  if( fOutStream) delete fOutStream;

}

void FlatEGunASCIIWriter::beginRun(const edm::Run &r, const EventSetup& es)
{

  es.getData( fPDGTable );

}

void FlatEGunASCIIWriter::beginJob()
{ 
  fOutStream = new HepMC::IO_GenEvent( fOutFileName.c_str() ); 
  if ( fOutStream->rdstate() == std::ios::failbit ) {
    throw cms::Exception("FileNotOpen", "FlatEGunASCIIWriter::beginJob()")
      << "File " << fOutFileName << " was not open.\n";
  }
  
  return ;

}

void FlatEGunASCIIWriter::analyze( const Event& , 
                                   const EventSetup& /* not used so far, thus is "empty" */ ) 
{
         
   // clean up GenEvent memory : also deletes all vtx/part in it
   // 
   if ( fEvt != NULL ) delete fEvt ;
   
   // here re-create fEvt (memory)
   fEvt = new HepMC::GenEvent() ;

   // now actualy, cook up the event from PDGTable and gun parameters
   //
   //HepMC::GenVertex* Vtx = new HepMC::GenVertex( CLHEP::HepLorentzVector(0.,0.,0.) );
   HepMC::GenVertex* Vtx = new HepMC::GenVertex( HepMC::FourVector(0.,0.,0.) );

   // loop over particles
   //
   for ( unsigned int ip=0; ip<fPartIDs.size(); ip++ )
   {
       double energy = CLHEP::RandFlat::shoot( fMinE, fMaxE ) ;
       double eta    = CLHEP::RandFlat::shoot( fMinEta, fMaxEta ) ;
       double phi    = CLHEP::RandFlat::shoot( fMinPhi, fMaxPhi ) ;
       const HepPDT::ParticleData* 
          PData = fPDGTable->particle(HepPDT::ParticleID(abs(fPartIDs[ip]))) ;
       double mass   = PData->mass().value() ;
       double mom2   = energy*energy - mass*mass ;
       double mom    = 0. ;
       if ( mom2 > 0. ) mom = sqrt(mom2) ;
       double theta  = 2.*atan(exp(-eta)) ;
       double px     = mom*sin(theta)*cos(phi) ;
       double py     = mom*sin(theta)*sin(phi) ;
       double pz     = mom*cos(theta) ;
       //CLHEP::Hep3Vector p(px,py,pz) ;
       //HepMC::GenParticle* Part = 
       //    new HepMC::GenParticle(CLHEP::HepLorentzVector(p,energy),fPartIDs[ip],1);
       HepMC::FourVector p(px,py,pz,energy);
       HepMC::GenParticle* Part = 
           new HepMC::GenParticle(p,fPartIDs[ip],1);
       Vtx->add_particle_out(Part);
   }
   fEvt->add_vertex( Vtx ) ;
   fEvt->set_event_number( fCurrentEvent+1 ) ;
   fEvt->set_signal_process_id(20) ; 
   
   // for testing purpose only
   // fEvt->print() ;     

   fOutStream->write_event( fEvt );

   fCurrentEvent++ ;

   return ;
   
}

