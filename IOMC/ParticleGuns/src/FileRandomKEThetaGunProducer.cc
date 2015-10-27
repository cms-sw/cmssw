#include <iostream>

#include "IOMC/ParticleGuns/interface/FileRandomKEThetaGunProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandomEngine.h"

using namespace edm;

FileRandomKEThetaGunProducer::FileRandomKEThetaGunProducer(const edm::ParameterSet& pset) :
  FlatBaseThetaGunProducer(pset) {

  edm::ParameterSet defpset ;
  edm::ParameterSet pgun_params = 
    pset.getParameter<edm::ParameterSet>("PGunParameters") ;
  
  edm::FileInPath fp   = pgun_params.getParameter<edm::FileInPath>("File");
  std::string     file = fp.fullPath();
  particleN            = pgun_params.getParameter<int>("Particles");
  if (particleN <= 0) particleN = 1;
  edm::LogInfo("FlatThetaGun") << "Internal FileRandomKEThetaGun is initialzed"
			       << " with data read from " << file << " and "
			       << particleN << " particles created/event";
  std::ifstream is(file.c_str(), std::ios::in);
  if (is) {
    double energy, elem, sum=0;
    while (!is.eof()) {
      is >> energy >> elem;
      kineticE.push_back(0.001*energy);
      fdistn.push_back(elem);
      sum += elem;
      if (fVerbosity > 0) LogDebug("FlatThetaGun") << "KE " << energy
						   <<" GeV Count rate " <<elem;
    }
    is.close();
    double last = 0;
    for (unsigned int i=0; i<fdistn.size(); i++) {
      fdistn[i] /= sum;
      fdistn[i] += last;
      last       = fdistn[i];
      if (fVerbosity > 0) LogDebug("FlatThetaGun") << "Bin [" << i << "]: KE "
						   << kineticE[i] << " Distn "
						   << fdistn[i];
    }
  }
  if (kineticE.size() < 2) 
    throw cms::Exception("FileNotFound","Not enough data point found in file ")
      <<  file << "\n";

  
  produces<HepMCProduct>("unsmeared");
  produces<GenEventInfoProduct>();
}

FileRandomKEThetaGunProducer::~FileRandomKEThetaGunProducer() {}

void FileRandomKEThetaGunProducer::produce(edm::Event & e, const edm::EventSetup& es) {

  if (fVerbosity > 0) 
    LogDebug("FlatThetaGun") << "FileRandomKEThetaGunProducer : Begin New Event Generation"; 
   
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

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
  for (int ip=0; ip<particleN; ip++) {
    double keMin=kineticE[0], keMax=kineticE[1];
    double rMin=fdistn[0], rMax=fdistn[1];
    double r1 = engine->flat();
    for (unsigned int ii=kineticE.size()-2; ii>0; --ii) {
      if (r1 > fdistn[ii]) {
	keMin = kineticE[ii]; keMax = kineticE[ii+1]; 
	rMin  = fdistn[ii];   rMax  = fdistn[ii+1]; break;
      }
    }
    double ke    = (keMin*(rMax-r1) + keMax*(r1-rMin))/(rMax-rMin);
    if (fVerbosity > 1) 
      LogDebug("FlatThetaGun") << "FileRandomKEThetaGunProducer: KE " << ke
			       << " in range " << keMin << ":" << keMax 
			       << " with " << r1 <<" in " << rMin <<":" <<rMax;
    double theta = CLHEP::RandFlat::shoot(engine, fMinTheta, fMaxTheta);
    double phi   = CLHEP::RandFlat::shoot(engine, fMinPhi, fMaxPhi);
    int PartID   = fPartIDs[0];
    const HepPDT::ParticleData* 
      PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID))) ;
    double mass   = PData->mass().value() ;
    double energy = ke + mass;
    double mom2   = ke*ke + 2.*ke*mass ;
    double mom    = std::sqrt(mom2);
    double px     = mom*sin(theta)*cos(phi);
    double py     = mom*sin(theta)*sin(phi);
    double pz     = mom*cos(theta);

    HepMC::FourVector p(px,py,pz,energy) ;
    HepMC::GenParticle* Part =  new HepMC::GenParticle(p,PartID,1);
    Part->suggest_barcode( barcode ) ;
    barcode++ ;
    Vtx->add_particle_out(Part);
       
  }
  fEvt->add_vertex(Vtx) ;
  fEvt->set_event_number(e.id().event()) ;
  fEvt->set_signal_process_id(20) ;  
   
   
  if (fVerbosity > 0) {
    fEvt->print() ;  
  }  

  std::auto_ptr<HepMCProduct> BProduct(new HepMCProduct()) ;
  BProduct->addHepMCData( fEvt );
  e.put(BProduct, "unsmeared");

  std::auto_ptr<GenEventInfoProduct> genEventInfo(new GenEventInfoProduct(fEvt));
  e.put(genEventInfo);

  if ( fVerbosity > 0 ) 
    LogDebug("FlatThetaGun") << "FileRandomKEThetaGunProducer : Event Generation Done";
}
