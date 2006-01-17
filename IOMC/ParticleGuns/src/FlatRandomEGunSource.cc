/*
 *  $Date: 2006/01/17 01:33:44 $
 *  $Revision: 1.2 $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomEGunSource.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "CLHEP/Random/RandFlat.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;
using namespace std;

FlatRandomEGunSource::FlatRandomEGunSource(const ParameterSet& pset,
                                           const InputSourceDescription& desc) :
   BaseFlatGunSource(pset, desc)
{


  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  
  fMinE = pgun_params.getParameter<double>("MinE");
  fMaxE = pgun_params.getParameter<double>("MaxE");
  
  //
  produces<HepMCProduct>();

  cout << "Internal FlatRandomEGun is initialzed" << endl ;
  cout << "It is going to generate " << remainingEvents() << "events" << endl ;
   
}

FlatRandomEGunSource::~FlatRandomEGunSource()
{
  if (fEvt != NULL) delete fEvt ;
  fEvt = 0 ;
}

bool FlatRandomEGunSource::produce(Event & e) 
{

   // event loop (well, another step in it...)
          
   // clean up GenEvent memory : also deletes all vtx/part in it
   // 
   if (fEvt != NULL) delete fEvt ;
   
   // here re-create fEvt (memory)
   fEvt = new HepMC::GenEvent() ;
   
   // now actualy, cook up the event from PDGTable and gun parameters
   //
   HepMC::GenVertex* Vtx = new HepMC::GenVertex(CLHEP::HepLorentzVector(0.,0.,0.));

   // loop over particles
   //
   for (unsigned int ip=0; ip<fPartIDs.size(); ip++)
   {
       double energy = RandFlat::shoot(fMinE, fMaxE) ;
       double eta    = RandFlat::shoot(fMinEta, fMaxEta) ;
       double phi    = RandFlat::shoot(fMinPhi, fMaxPhi) ;
       DefaultConfig::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(abs(fPartIDs[ip]))) ;
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
    
   // for testing purpose only
   // fEvt->print() ;
   // cout << " FlatRandomEGunSource : Event Generation Done " << endl;
      
   return true;
}
