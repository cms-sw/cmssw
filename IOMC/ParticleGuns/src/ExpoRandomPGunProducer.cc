/*
 *  \author Jean-Roch Vlimant
 *  modified by S.Abdullin 04/02/2011 
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/ExpoRandomPGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

ExpoRandomPGunProducer::ExpoRandomPGunProducer(const ParameterSet& pset) :
   BaseFlatGunProducer(pset)
{


   ParameterSet defpset ;
   ParameterSet pgun_params =
      pset.getParameter<ParameterSet>("PGunParameters") ;

   fMinP = pgun_params.getParameter<double>("MinP");
   fMaxP = pgun_params.getParameter<double>("MaxP");

   produces<HepMCProduct>("unsmeared");
   produces<GenEventInfoProduct>();

}

ExpoRandomPGunProducer::~ExpoRandomPGunProducer()
{
   // no need to cleanup GenEvent memory - done in HepMCProduct                                                               
}

void ExpoRandomPGunProducer::produce(Event &e, const EventSetup& es)
{
   edm::Service<edm::RandomNumberGenerator> rng;
   CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());
  
   if ( fVerbosity > 0 )
   {
     std::cout << " ExpoRandomPGunProducer : Begin New Event Generation"
               << std::endl ;
   }
   // event loop (well, another step in it...)                           

   // no need to clean up GenEvent memory - done in HepMCProduct         
   //

   // here re-create fEvt (memory)                                       
   //
                                             
   fEvt = new HepMC::GenEvent() ;

   // now actualy, cook up the event from PDGTable and gun parameters
   //
   // 1st, primary vertex
   //
   //HepMC::GenVertex* Vtx = new HepMC::GenVertex(CLHEP::HepLorentzVector(0.,0.,0.));                                         
   HepMC::GenVertex* Vtx = new HepMC::GenVertex(HepMC::FourVector(0.,0.,0.));

   // loop over particles
   //
   int barcode = 1 ;
   for (unsigned int ip=0; ip<fPartIDs.size(); ++ip)
   {

     double pmom   = CLHEP::RandFlat::shoot(engine, fMinP, fMaxP);
     double y      = (1./fMinP) * CLHEP::RandFlat::shoot(engine, 0.0, 1.0);
     double f      = 1./pmom;
     bool   accpt  = ( y < f);
     //shoot until in the designated range                             
     while ((pmom < fMinP || pmom > fMaxP) || !accpt)
       {
         pmom   = CLHEP::RandFlat::shoot(engine, fMinP, fMaxP);
         y      = (1./fMinP) * CLHEP::RandFlat::shoot(engine, 0.0, 1.0);
         f      = 1./pmom;
         accpt  = (y < f);
	 
       }

       double eta    = CLHEP::RandFlat::shoot(engine, fMinEta, fMaxEta) ;
       double phi    = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi) ;
       int PartID = fPartIDs[ip] ;
       const HepPDT::ParticleData*
          PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID))) ;
       double mass   = PData->mass().value() ;
       double theta  = 2.*atan(exp(-eta)) ;                  
       double mom    = pmom;
       double pt     = mom * sin(theta);
       double px     = pt  * cos(phi) ;
       double py     = pt  * sin(phi) ;          
       double pz     = mom*cos(theta) ;
       double energy2= mom*mom + mass*mass ;
       double energy = sqrt(energy2) ;
       //CLHEP::Hep3Vector p(px,py,pz) ;
       //HepMC::GenParticle* Part =
       //    new HepMC::GenParticle(CLHEP::HepLorentzVector(p,energy),PartID,1);                                              
       HepMC::FourVector p(px,py,pz,energy) ;
       HepMC::GenParticle* Part =
           new HepMC::GenParticle(p,PartID,1);
       Part->suggest_barcode( barcode ) ;
       barcode++ ;
       Vtx->add_particle_out(Part);

       if ( fAddAntiParticle )
       {
          //CLHEP::Hep3Vector ap(-px,-py,-pz) ;
          HepMC::FourVector ap(-px,-py,-pz,energy) ;
          int APartID = -PartID ;
          if ( PartID == 22 || PartID == 23 )
          {
             APartID = PartID ;
          }
          //HepMC::GenParticle* APart =
          //   new HepMC::GenParticle(CLHEP::HepLorentzVector(ap,energy),APartID,1);                                          
          HepMC::GenParticle* APart =
             new HepMC::GenParticle(ap,APartID,1);
          APart->suggest_barcode( barcode ) ;
          barcode++ ;
          Vtx->add_particle_out(APart) ;
       }

   }

   fEvt->add_vertex(Vtx) ;
   fEvt->set_event_number(e.id().event()) ;
   fEvt->set_signal_process_id(20) ;

   if ( fVerbosity > 0 )
   {
      fEvt->print() ;
   }

   auto_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
   BProduct->addHepMCData( fEvt );
   e.put(BProduct, "unsmeared");

   auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
   e.put(genEventInfo);

   if ( fVerbosity > 0 )
   {
      // for testing purpose only
      // fEvt->print() ; // prints empty info after it's made into edm::Event                                                 
     std::cout << " FlatRandomPGunProducer : Event Generation Done " << std::endl;
   }
}
