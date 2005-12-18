/*
 *  $Date: $
 *  $Revision: $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomEGunSource.h"

#include "FWCore/Framework/src/TypeID.h" 
#include "PluginManager/ModuleDef.h"

#include "CLHEP/Random/RandFlat.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;
using namespace std;

FlatRandomEGunSource::FlatRandomEGunSource( const ParameterSet& pset,
                                            const InputSourceDescription& desc ) : 
   BaseFlatGunSource( pset, desc )
{


  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  
  fMinE = pgun_params.getParameter<double>("MinE");
  fMaxE = pgun_params.getParameter<double>("MaxE");
  
  // now do the "unique" stuff
  //
  ModuleDescription      ModDesc; 
  ModDesc.pid            = PS_ID("FlatRandomEGunSource");
  ModDesc.moduleName_    = "FlatRandomEGunSource";
  ModDesc.moduleLabel_   = "FlatRandomEGun";
  ModDesc.versionNumber_ = 1UL;
  ModDesc.processName_   = "HepMC";
  ModDesc.pass           = 1UL;  
      
  fBranchDesc.module             = ModDesc ;
  fBranchDesc.fullClassName_     = "HepMCProduct" ;
  fBranchDesc.friendlyClassName_ = "HepMCProduct" ;
  
  // and register it (with BaseFlatGunSource, in fact !)
  //
  registerBranch( fBranchDesc ) ;

  cout << "Internal FlatRandomEGun is initialzed" << endl ;
  cout << "It is going to generate " << fNEventsToProcess << "events" << endl ;
   
}

FlatRandomEGunSource::~FlatRandomEGunSource()
{
  if ( fEvt != NULL ) delete fEvt ;
  fEvt = 0 ;
}

auto_ptr<EventPrincipal> FlatRandomEGunSource::read() 
{

   // 0-result
   auto_ptr<EventPrincipal> Result(0);
  
   // we generate here while NEvents < Max
   if ( fCurrentEvent >= fNEventsToProcess ) return Result ; // 0-result will terminate the loop  
   
   // event loop (well, another step in it...)
          
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
       DefaultConfig::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(fPartIDs[ip])) ;
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


   Result = insertHepMCEvent( fBranchDesc ) ;
    
   fNextID = fNextID.next();
   fNextTime += fTimeBetweenEvents;

   // for testing purpose only
   // fEvt->print() ;
   // cout << " FlatRandomEGunSource : Event Generation Done " << endl;
      
   fCurrentEvent++ ;
        
   return Result;

}
      
 

