/*
 *  $Date: 2006/04/07 04:04:20 $
 *  $Revision: 1.7 $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomEGunSource.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;
using namespace std;

FlatRandomEGunSource::FlatRandomEGunSource(const ParameterSet& pset,
                                           const InputSourceDescription& desc) :
   BaseFlatGunSource(pset, desc)
{

   ParameterSet defpset ;
   // ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
   ParameterSet pgun_params = 
      pset.getUntrackedParameter<ParameterSet>("PGunParameters",defpset) ;
  
   // doesn't seem necessary to check if pset is empty - if this
   // is the case, default values will be taken for params
   fMinE = pgun_params.getUntrackedParameter<double>("MinE",0.99);
   fMaxE = pgun_params.getUntrackedParameter<double>("MaxE",1.01);

   cout << "NParts = " << fPartIDs.size() << endl ;
   for ( int i=0; i<fPartIDs.size(); i++ )
   {
      cout << fPartIDs[i] << endl ;
   }
   cout << " MinEta, MaxEta = " << fMinEta << " " << fMaxEta << endl ;
   cout << " MinPhi, MaxPhi = " << fMinPhi << " " << fMaxPhi << endl ;
   cout << " MinE, MaxE = " << fMinE << " " << fMaxE << endl ;
  
  
  produces<HepMCProduct>();

  cout << "Internal FlatRandomEGun is initialzed" << endl ;
  cout << "It is going to generate " << remainingEvents() << "events" << endl ;
   
}

FlatRandomEGunSource::~FlatRandomEGunSource()
{
   // no need to cleanup fEvt since it's done in HepMCProduct
}

bool FlatRandomEGunSource::produce(Event & e) 
{

   if ( fVerbosity > 0 )
   {
      cout << " FlatRandomEGunSource : Begin New Event Generation" << endl ; 
   }
   
   // event loop (well, another step in it...)
          
   // no need to clean up GenEvent memory - done in HepMCProduct

   // here re-create fEvt (memory)
   //
   fEvt = new HepMC::GenEvent() ;
   
   // now actualy, cook up the event from PDGTable and gun parameters
   //

   // 1st, primary vertex
   //
   HepMC::GenVertex* Vtx = new HepMC::GenVertex(CLHEP::HepLorentzVector(0.,0.,0.));
   
   // loop over particles
   //
   for (unsigned int ip=0; ip<fPartIDs.size(); ip++)
   {
       //double energy = RandFlat::shoot(fMinE, fMaxE) ;
       //double eta    = RandFlat::shoot(fMinEta, fMaxEta) ;
       //double phi    = RandFlat::shoot(fMinPhi, fMaxPhi) ;
       double energy = fRandomGenerator->fire(fMinE, fMaxE) ;
       double eta    = fRandomGenerator->fire(fMinEta, fMaxEta) ;
       double phi    = fRandomGenerator->fire(fMinPhi, fMaxPhi) ;
       DefaultConfig::ParticleData* 
          PData = fPDGTable->particle(HepPDT::ParticleID(abs(fPartIDs[ip]))) ;
       double mass   = PData->mass().value() ;
       double mom2   = energy*energy - mass*mass ;
       double mom    = 0. ;
       if (mom2 > 0.) 
       {
          mom = sqrt(mom2) ;
       }
       else
       {
          mom = 0. ;
       }
       double theta  = 2.*atan(exp(-eta)) ;
       double px     = mom*sin(theta)*cos(phi) ;
       double py     = mom*sin(theta)*sin(phi) ;
       double pz     = mom*cos(theta) ;

       CLHEP::Hep3Vector p(px,py,pz) ;
       HepMC::GenParticle* Part = 
           new HepMC::GenParticle(CLHEP::HepLorentzVector(p,energy),fPartIDs[ip],1);
       Vtx->add_particle_out(Part);
   }
   fEvt->add_vertex(Vtx) ;
   fEvt->set_event_number(event()) ;
   fEvt->set_signal_process_id(20) ;      

   auto_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
   BProduct->addHepMCData( fEvt );
   e.put(BProduct);
    
   if ( fVerbosity > 0 )
   {
      // for testing purpose only
      fEvt->print() ;
      cout << " FlatRandomEGunSource : Event Generation Done " << endl;
   }
      
   return true;
}
