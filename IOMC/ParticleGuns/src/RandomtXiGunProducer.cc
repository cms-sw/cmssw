/*
 *  $Date: 2011/12/19 23:10:40 $
 *  $Revision: 1.4 $
 *  \author Luiz Mundim
 */

#include <ostream>

#include "IOMC/ParticleGuns/interface/RandomtXiGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

RandomtXiGunProducer::RandomtXiGunProducer(const edm::ParameterSet& pset) : 
   BaseRandomtXiGunProducer(pset)
{

   ParameterSet defpset ;
   edm::ParameterSet pgun_params = 
      pset.getParameter<edm::ParameterSet>("PGunParameters") ;
  
   fMint = pgun_params.getParameter<double>("Mint");
   fMaxt = pgun_params.getParameter<double>("Maxt");
   fMinXi= pgun_params.getParameter<double>("MinXi");
   fMaxXi= pgun_params.getParameter<double>("MaxXi");
  
  produces<HepMCProduct>("unsmeared");
  produces<GenEventInfoProduct>();
}

RandomtXiGunProducer::~RandomtXiGunProducer()
{
   // no need to cleanup GenEvent memory - done in HepMCProduct
}

void RandomtXiGunProducer::produce(edm::Event &e, const edm::EventSetup& es) 
{
   edm::Service<edm::RandomNumberGenerator> rng;
   CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

   if ( fVerbosity > 0 )
   {
      cout << " RandomtXiGunProducer : Begin New Event Generation" << endl ; 
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
   HepMC::GenVertex* Vtx = new HepMC::GenVertex(HepMC::FourVector(0.,0.,0.));

   // loop over particles
   //
   int barcode = 1 ;
   for (unsigned int ip=0; ip<fPartIDs.size(); ++ip)
   {
       int PartID = fPartIDs[ip];
//  t = -2*P*P'*(1-cos(theta)) -> t/(2*P*P')+1=cos(theta)
// xi = 1 - P'/P  --> P'= (1-xi)*P
// 
       PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID)));
       if (!PData) exit(1);
       double t  = 0;
       double Xi = 0;
       double phi=0;
       if (fFireForward) {
          while(true) {
               Xi     = CLHEP::RandFlat::shoot(engine,fMinXi,fMaxXi);
               double min_t = std::max(fMint,Minimum_t(Xi));
               double max_t = fMaxt;
               if (min_t>max_t) {
                  std::cout << "WARNING: t limits redefined (unphysical values for given xi)." << endl;
                  max_t = min_t;
               }
               t      = CLHEP::RandFlat::shoot(engine,min_t,max_t);
               break;
          }
          phi    = CLHEP::RandFlat::shoot(engine,fMinPhi, fMaxPhi) ;
          HepMC::GenParticle* Part = 
              new HepMC::GenParticle(make_particle(t,Xi,phi,PartID,1),PartID,1);
          Part->suggest_barcode( barcode ) ;
          barcode++ ;
          Vtx->add_particle_out(Part);
       }
       if ( fFireBackward) {
          while(true) {
               Xi     = CLHEP::RandFlat::shoot(engine,fMinXi,fMaxXi);
               double min_t = std::max(fMint,Minimum_t(Xi));
               double max_t = fMaxt;
               if (min_t>max_t) {
                  std::cout << "WARNING: t limits redefined (unphysical values for given xi)." << endl;
                  max_t = min_t;
               }
               t      = CLHEP::RandFlat::shoot(engine,min_t,max_t);
               break;
          }
          phi    = CLHEP::RandFlat::shoot(engine,fMinPhi, fMaxPhi) ;
	  HepMC::GenParticle* Part2 =
	     new HepMC::GenParticle(make_particle(t,Xi,phi,PartID,-1),PartID,1);
	  Part2->suggest_barcode( barcode ) ;
	  barcode++ ;
	  Vtx->add_particle_out(Part2) ;
       }
   }

   fEvt->add_vertex(Vtx) ;
   fEvt->set_event_number(e.id().event()) ;
   fEvt->set_signal_process_id(20) ; 
        
   if ( fVerbosity > 0 )
   {
      fEvt->print() ;  
   }

   std::unique_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
   BProduct->addHepMCData( fEvt );
   e.put(std::move(BProduct),"unsmeared");

   std::unique_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
   e.put(std::move(genEventInfo));
    
   if ( fVerbosity > 0 )
   {
      // for testing purpose only
      // fEvt->print() ; // prints empty info after it's made into edm::Event
      cout << " RandomtXiGunProducer : Event Generation Done " << endl;
   }
}
HepMC::FourVector RandomtXiGunProducer::make_particle(double t,double Xi,double phi,int PartID, int direction)
{
       double mass   = PData->mass().value() ;
       double fpMom  = sqrt(fpEnergy*fpEnergy-mass*mass); // momentum of beam proton
       double sEnergy = (1.0-Xi)*fpEnergy; // energy of scattered particle
       double sMom    = sqrt(sEnergy*sEnergy-mass*mass); // momentum of scattered particle
       double min_t = -2.*(fpMom*sMom-fpEnergy*sEnergy+mass*mass);
       if (t<min_t) t=min_t; // protect against kinemactically forbiden region
       long double theta  = acos((-t/2.- mass*mass + fpEnergy*sEnergy)/(sMom*fpMom)); // use t = -t

       if (direction<1) theta = acos(-1.) - theta;

       double px     = sMom*cos(phi)*sin(theta)*direction;
       double py     = sMom*sin(phi)*sin(theta);
       double pz     = sMom*cos(theta) ;  // the direction is already set by the theta angle
       if (fVerbosity > 0) 
          edm::LogInfo("RandomXiGunProducer") << "-----------------------------------------------------------------------------------------------------\n"
                                              << "Produced a proton with  phi : " << phi << " theta: " << theta << " t: " << t << " Xi: " << Xi << "\n"
                                              << "                         Px : " << px  << " Py : " << py << " Pz : " << pz << "\n"
                                              << "                   direction: " << direction  << "\n"
                                              << "-----------------------------------------------------------------------------------------------------"
                                              << std::endl;

       return HepMC::FourVector(px,py,pz,sEnergy) ;
}
//#include "FWCore/Framework/interface/MakerMacros.h"
//DEFINE_FWK_MODULE(RandomtXiGunProducer);
