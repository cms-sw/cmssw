/*
 *  $Date: 2006/01/17 23:17:25 $
 *  $Revision: 1.3 $
 *  \author Julia Yarba
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomPtGunSource.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

FlatRandomPtGunSource::FlatRandomPtGunSource(const ParameterSet& pset,
                                             const InputSourceDescription& desc) : 
   BaseFlatGunSource(pset, desc)
{


  ParameterSet pgun_params = pset.getParameter<ParameterSet>("PGunParameters") ;
  
  fMinPt = pgun_params.getParameter<double>("MinPt");
  fMaxPt = pgun_params.getParameter<double>("MaxPt");
  
  produces<HepMCProduct>();

  cout << "Internal FlatRandomPtGun is initialzed" << endl ;
  cout << "It is going to generate " << remainingEvents() << " events" << endl ;
   
}

FlatRandomPtGunSource::~FlatRandomPtGunSource()
{
  if (fEvt != NULL) delete fEvt ;
  fEvt = 0 ;
}

bool FlatRandomPtGunSource::produce(Event &e) 
{

   if ( fVerbosity > 0 )
   {
      cout << " FlatRandomPtGunSource : Begin New Event Generation" << endl ; 
   }
   // event loop (well, another step in it...)
          
   // clean up GenEvent memory : also deletes all vtx/part in it
   // 
   if (fEvt != NULL) delete fEvt ;
   
   // here re-create fEvt (memory)
   fEvt = new HepMC::GenEvent() ;
   
   // now actualy, cook up the event from PDGTable and gun parameters
   //
   // 1st, primary vertex
   //
   // HepMC::GenVertex* Vtx = new HepMC::GenVertex(CLHEP::HepLorentzVector(0.,0.,0.));
   HepMC::GenVertex* Vtx = generateEvtVertex() ;
      
   if ( fVerbosity > 0 )
   {
      cout << " Vtx = " << Vtx->position().x() << " " 
                        << Vtx->position().y() << " " 
		        << Vtx->position().z() << endl ;
   }

   // loop over particles
   //
   for (unsigned int ip=0; ip<fPartIDs.size(); ++ip)
   {

       double pt     = RandFlat::shoot(fMinPt, fMaxPt) ;
       double eta    = RandFlat::shoot(fMinEta, fMaxEta) ;
       double phi    = RandFlat::shoot(fMinPhi, fMaxPhi) ;
       DefaultConfig::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(abs(fPartIDs[ip]))) ;
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
      cout << " FlatRandomPtGunSource : Event Generation Done " << endl;
   }

   return true;
}

