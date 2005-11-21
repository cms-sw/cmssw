/*
 *  $Date: $
 *  $Revision: $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/Input/interface/FlatRandomEGunSource.h"

#include "FWCore/Framework/src/TypeID.h" 

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include <FWCore/EDProduct/interface/Wrapper.h>

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

//used for defaults
  static const unsigned long kNanoSecPerSec = 1000000000;
  static const unsigned long kAveEventPerSec = 200;

FlatRandomEGunSource::FlatRandomEGunSource( const ParameterSet& pset,
                                            const InputSourceDescription& desc ) : 
  InputSource ( desc ),
  fNEventsToProcess(pset.getUntrackedParameter<int>("maxEvents", -1)),
  fCurrentEvent(0), 
  fCurrentRun( pset.getUntrackedParameter<unsigned int>("firstRun",1)  ),
  fNextTime(pset.getUntrackedParameter<unsigned int>("firstTime",1)),  //time in ns
  fTimeBetweenEvents(pset.getUntrackedParameter<unsigned int>("timeBetweenEvents",kNanoSecPerSec/kAveEventPerSec) ),
  fEvt(0),
  fPDGTable( new DefaultConfig::ParticleDataTable("PDG Table") ),
  fNextID(fCurrentRun, 1 ) 
{

  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  fPartIDs    = pgun_params.getParameter< vector<int> >("PartID");
  fMinEta     = pgun_params.getParameter<double>("MinEta");
  fMaxEta     = pgun_params.getParameter<double>("MaxEta");
  fMinPhi     = pgun_params.getParameter<double>("MinPhi");
  fMaxPhi     = pgun_params.getParameter<double>("MaxPhi");
  fMinE       = pgun_params.getParameter<double>("MinE");
  fMaxE       = pgun_params.getParameter<double>("MaxE");
  
  // hardcoded for now
  fPDGTablePath = "/afs/cern.ch/sw/lcg/external/clhep/1.9.2.1/slc3_ia32_gcc323/data/HepPDT/" ;
  fPDGTableName = "PDG_mass_width_2002.mc";

  string TableFullName = fPDGTablePath + fPDGTableName ;
  ifstream PDFile( TableFullName.c_str() ) ;
  if( !PDFile ) 
  {
      cerr << "cannot open " << TableFullName << endl;
      exit(-1);
  }

  HepPDT::TableBuilder tb(*fPDGTable) ;
  if ( !addPDGParticles( PDFile, tb ) ) { cout << " Error reading PDG !" << endl; }
  // the tb dtor fills fPDGTable

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
  
  preg_->addProduct( fBranchDesc ) ;
  
  cout << "Internal FlatRandomEGun is initialzed" << endl ;
  cout << "It is going to generate " << fNEventsToProcess << "events" << endl ;
   
}


FlatRandomEGunSource::~FlatRandomEGunSource()
{
  if ( fEvt != NULL ) delete fEvt ;
  delete fPDGTable;
}

auto_ptr<EventPrincipal> FlatRandomEGunSource::read() 
{

   // 0-result
   auto_ptr<EventPrincipal> Result(0);
  
   // we generate here while NEvents < Max
   if ( fCurrentEvent >= fNEventsToProcess ) return Result ; // 0-result will terminate the loop  
   
  // event loop (well, another step in it...)
      
   // for testing purpose
   // cout << "FlatRandomEGunSource : Create new Particle(s)/Event " << endl;
    
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
   Result = auto_ptr<EventPrincipal>(new EventPrincipal(fNextID, 
                                                        Timestamp(fNextTime),  
	 					       // *fRetriever, 
							*preg_) ) ;							 
							
   if(fEvt)  
   {
       auto_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
       BProduct->addHepMCData( fEvt );
       edm::Wrapper<HepMCProduct>* WProduct = 
            new edm::Wrapper<HepMCProduct>(BProduct); 
       auto_ptr<EDProduct>  FinalProduct(WProduct);
       auto_ptr<Provenance> Prov(new Provenance(fBranchDesc)) ;
       Result->put(FinalProduct, Prov);
   }
    
   fNextID = fNextID.next();
   fNextTime += fTimeBetweenEvents;

   // for testing purpose only
   // fEvt->print() ;
   // cout << " FlatRandomEGunSource : Event Generation Done " << endl;

   fCurrentEvent++ ;
     
   return Result;

}
      
 

