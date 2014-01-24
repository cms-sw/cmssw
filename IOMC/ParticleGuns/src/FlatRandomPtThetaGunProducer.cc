#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomPtThetaGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"

namespace CLHEP {
  class HepRandomEngine;
}

using namespace edm;

FlatRandomPtThetaGunProducer::FlatRandomPtThetaGunProducer(const edm::ParameterSet& pset) :
  FlatBaseThetaGunProducer(pset)
{
  edm::ParameterSet defpset ;
  edm::ParameterSet pgun_params = 
    pset.getParameter<edm::ParameterSet>("PGunParameters") ;
  
  fMinPt = pgun_params.getParameter<double>("MinPt");
  fMaxPt = pgun_params.getParameter<double>("MaxPt");
  
  produces<HepMCProduct>();
  produces<GenEventInfoProduct>();
//  edm::LogInfo("FlatThetaGun") << "Internal FlatRandomPtThetaGun is initialzed"
//			       << "\nIt is going to generate " 
//			       << remainingEvents() << " events";
}

FlatRandomPtThetaGunProducer::~FlatRandomPtThetaGunProducer() {}

void FlatRandomPtThetaGunProducer::produce(edm::Event &e, const EventSetup& es) {

  if ( fVerbosity > 0 ) {
    LogDebug("FlatThetaGun") << "FlatRandomPtThetaGunProducer : Begin New Event Generation"; 
  }

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

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
  for (unsigned int ip=0; ip<fPartIDs.size(); ++ip) {

    double pt     = CLHEP::RandFlat::shoot(engine, fMinPt, fMaxPt);
    double theta  = CLHEP::RandFlat::shoot(engine, fMinTheta, fMaxTheta);
    double phi    = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi);
    int PartID = fPartIDs[ip] ;
    const HepPDT::ParticleData* 
      PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID))) ;
    double mass   = PData->mass().value() ;
    double mom    = pt/sin(theta) ;
    double px     = pt*cos(phi) ;
    double py     = pt*sin(phi) ;
    double pz     = mom*cos(theta) ;
    double energy2= mom*mom + mass*mass ;
    double energy = sqrt(energy2) ; 
    HepMC::FourVector p(px,py,pz,energy) ;
    HepMC::GenParticle* Part = new HepMC::GenParticle(p,PartID,1);
    Part->suggest_barcode( barcode ) ;
    barcode++ ;
    Vtx->add_particle_out(Part);

    if ( fAddAntiParticle ) {
      HepMC::FourVector ap(-px,-py,-pz,energy) ;
      int APartID = -PartID ;
      if ( PartID == 22 || PartID == 23 ) {
	APartID = PartID ;
      }	  
      HepMC::GenParticle* APart = new HepMC::GenParticle(ap,APartID,1);
      APart->suggest_barcode( barcode ) ;
      barcode++ ;
      Vtx->add_particle_out(APart) ;
    }
  }

  fEvt->add_vertex(Vtx) ;
  fEvt->set_event_number(e.id().event()) ;
  fEvt->set_signal_process_id(20) ; 
        
  if ( fVerbosity > 0 ) {
    fEvt->print() ;  
  }

  std::auto_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
  BProduct->addHepMCData( fEvt );
  e.put(BProduct);

  std::auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
  e.put(genEventInfo);

  if ( fVerbosity > 0 ) {
    LogDebug("FlatThetaGun") << "FlatRandomPtThetaGunProducer : Event Generation Done ";
  }
}
