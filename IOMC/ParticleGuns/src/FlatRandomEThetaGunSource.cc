#include <ostream>

#include "IOMC/ParticleGuns/interface/FlatRandomEThetaGunSource.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;

FlatRandomEThetaGunSource::FlatRandomEThetaGunSource(const edm::ParameterSet& pset,
						     const edm::InputSourceDescription& desc) :
  FlatBaseThetaGunSource(pset, desc) {

  edm::ParameterSet defpset ;
  edm::ParameterSet pgun_params = 
    pset.getUntrackedParameter<edm::ParameterSet>("PGunParameters",defpset) ;
  
  // doesn't seem necessary to check if pset is empty - if this
  // is the case, default values will be taken for params
  fMinE = pgun_params.getUntrackedParameter<double>("MinE",0.99);
  fMaxE = pgun_params.getUntrackedParameter<double>("MaxE",1.01); 
  
  produces<HepMCProduct>();

  edm::LogInfo("FlatThetaGun") << "Internal FlatRandomEThetaGun is initialzed"
			       << "\nIt is going to generate " 
			       << remainingEvents() << " events";
}

FlatRandomEThetaGunSource::~FlatRandomEThetaGunSource() {}

bool FlatRandomEThetaGunSource::produce(edm::Event & e) {

  if ( fVerbosity > 0 ) {
    LogDebug("FlatThetaGun") << "FlatRandomEThetaGunSource : Begin New Event Generation"; 
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
  HepMC::GenVertex* Vtx = new HepMC::GenVertex( HepMC::FourVector(0.,0.,0.));
   
  // loop over particles
  //
  int barcode = 1;
  for (unsigned int ip=0; ip<fPartIDs.size(); ip++) {
    double energy = fRandomGenerator->fire(fMinE, fMaxE) ;
    double theta  = fRandomGenerator->fire(fMinTheta, fMaxTheta) ;
    double phi    = fRandomGenerator->fire(fMinPhi, fMaxPhi) ;
    int PartID = fPartIDs[ip] ;
    const HepPDT::ParticleData* 
      PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID))) ;
    double mass   = PData->mass().value() ;
    double mom2   = energy*energy - mass*mass ;
    double mom    = (mom2 > 0. ? std::sqrt(mom2) : 0.);
    double px     = mom*sin(theta)*cos(phi) ;
    double py     = mom*sin(theta)*sin(phi) ;
    double pz     = mom*cos(theta) ;

    HepMC::FourVector p(px,py,pz,energy) ;
    HepMC::GenParticle* Part =  new HepMC::GenParticle(p,PartID,1);
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
  fEvt->set_event_number(event()) ;
  fEvt->set_signal_process_id(20) ;  
   
   
  if ( fVerbosity > 0 ) {
    fEvt->print() ;  
  }  

  std::auto_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
  BProduct->addHepMCData( fEvt );
  e.put(BProduct);
    
  if ( fVerbosity > 0 ) {
    LogDebug("FlatThetaGun") << "FlatRandomEThetaGunSource : Event Generation Done";
   }
  
  return true;
}
