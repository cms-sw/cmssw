/*
 *  $Date: $
 *  $Revision: $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomPtGunSource.h"

#include "FWCore/Framework/src/TypeID.h" 
#include "PluginManager/ModuleDef.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

FlatRandomPtGunSource::FlatRandomPtGunSource( const ParameterSet& pset,
                                              const InputSourceDescription& desc ) : 
   BaseFlatGunSource( pset, desc )
{


  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  
  fMinPt = pgun_params.getParameter<double>("MinPt");
  fMaxPt = pgun_params.getParameter<double>("MaxPt");
  
  // now do the "unique" stuff
  //
  ModuleDescription      ModDesc; 
  ModDesc.pid            = PS_ID("FlatRandomPtGunSource");
  ModDesc.moduleName_    = "FlatRandomPtGunSource";
  ModDesc.moduleLabel_   = "FlatRandomPtGun";
  ModDesc.versionNumber_ = 1UL;
  ModDesc.processName_   = "HepMC";
  ModDesc.pass           = 1UL;  
      
  fBranchDesc.module             = ModDesc ;
  fBranchDesc.fullClassName_     = "HepMCProduct" ;
  fBranchDesc.friendlyClassName_ = "HepMCProduct" ;
  
  // and register it (with BaseFlatGunSource, in fact !)
  //
  registerBranch( fBranchDesc ) ;

  cout << "Internal FlatRandomPtGun is initialzed" << endl ;
  cout << "It is going to generate " << fNEventsToProcess << "events" << endl ;
   
}

FlatRandomPtGunSource::~FlatRandomPtGunSource()
{
  if ( fEvt != NULL ) delete fEvt ;
  fEvt = 0 ;
}

auto_ptr<EventPrincipal> FlatRandomPtGunSource::read() 
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

       double pt     = RandFlat::shoot( fMinPt, fMaxPt ) ;
       double eta    = RandFlat::shoot( fMinEta, fMaxEta ) ;
       double phi    = RandFlat::shoot( fMinPhi, fMaxPhi ) ;
       DefaultConfig::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(fPartIDs[ip])) ;
       double mass   = PData->mass().value() ;
       double theta  = 2.*atan(exp(-eta)) ;
       double mom    = pt/sin(theta) ;
       double px     = pt*cos(phi) ;
       double py     = pt*sin(phi) ;
       double pz     = mom*cos(theta) ;
       double energy2= mom*mom + mass*mass ;
       double energy = sqrt(energy2) ; 
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
   // cout << " FlatRandomPtGunSource : Event Generation Done " << endl;
      
   fCurrentEvent++ ;
        
   return Result;

}
      
 

