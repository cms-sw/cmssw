#include <ostream>

#include "IOMC/ParticleGuns/interface/CloseByParticleGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/Math/interface/Vector3D.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Random/RandFlat.h"

using namespace edm;
using namespace std;

CloseByParticleGunProducer::CloseByParticleGunProducer(const ParameterSet& pset) :
   BaseFlatGunProducer(pset)
{

  ParameterSet defpset ;
  ParameterSet pgun_params =
    pset.getParameter<ParameterSet>("PGunParameters") ;

  fEn = pgun_params.getParameter<double>("En");
  fR = pgun_params.getParameter<double>("R");
  fZ = pgun_params.getParameter<double>("Z");
  fDelta = pgun_params.getParameter<double>("Delta");
  fPartIDs = pgun_params.getParameter< vector<int> >("PartID");
  fPointing = pgun_params.getParameter<bool>("Pointing");

  produces<HepMCProduct>("unsmeared");
  produces<GenEventInfoProduct>();
}

CloseByParticleGunProducer::~CloseByParticleGunProducer()
{
   // no need to cleanup GenEvent memory - done in HepMCProduct
}

void CloseByParticleGunProducer::produce(Event &e, const EventSetup& es)
{
   edm::Service<edm::RandomNumberGenerator> rng;
   CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

   if ( fVerbosity > 0 )
     {
       LogDebug("CloseByParticleGunProducer") << " CloseByParticleGunProducer : Begin New Event Generation" << endl ;
     }
   fEvt = new HepMC::GenEvent() ;

   // loop over particles
   //
   int barcode = 1 ;
   double phi = CLHEP::RandFlat::shoot(engine, -3.14159265358979323846, 3.14159265358979323846);
   for (unsigned int ip=0; ip<fPartIDs.size(); ++ip, phi += fDelta/fR)
   {

     int PartID = fPartIDs[ip] ;
     const HepPDT::ParticleData *PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID))) ;
     double mass   = PData->mass().value() ;
     double mom    = sqrt(fEn*fEn-mass*mass);
     double px     = 0.;
     double py     = 0.;
     double pz     = mom;
     double energy = fEn;

     // Compute Vertex Position
     double x=fR*cos(phi);
     double y=fR*sin(phi);
     constexpr double c= 2.99792458e+1; // cm/ns
     double timeOffset = sqrt(x*x + y*y + fZ*fZ)/c*ns*c_light;
     HepMC::GenVertex* Vtx = new HepMC::GenVertex(HepMC::FourVector(x*cm,y*cm,fZ*cm,timeOffset));

     HepMC::FourVector p(px,py,pz,energy) ;
     // If we are requested to be pointing to (0,0,0), correct the momentum direction
     if (fPointing) {
       math::XYZVector direction(x,y,fZ);
       math::XYZVector momentum = direction.unit() * mom;
       p.setX(momentum.x());
       p.setY(momentum.y());
       p.setZ(momentum.z());
     }
     HepMC::GenParticle* Part = new HepMC::GenParticle(p,PartID,1);
     Part->suggest_barcode( barcode );
     barcode++;

     Vtx->add_particle_out(Part);

     if (fVerbosity > 0) {
       Vtx->print();
       Part->print();
     }
     fEvt->add_vertex(Vtx);
   }


   fEvt->set_event_number(e.id().event());
   fEvt->set_signal_process_id(20);

   if ( fVerbosity > 0 )
   {
      fEvt->print();
   }

   unique_ptr<HepMCProduct> BProduct(new HepMCProduct());
   BProduct->addHepMCData( fEvt );
   e.put(std::move(BProduct), "unsmeared");

   unique_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
   e.put(std::move(genEventInfo));

   if ( fVerbosity > 0 )
     {
       LogDebug("CloseByParticleGunProducer") << " CloseByParticleGunProducer : Event Generation Done " << endl;
     }
}

