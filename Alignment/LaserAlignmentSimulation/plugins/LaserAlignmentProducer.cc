/** \file LaserAlignmentProducer.cc
 *  Producer to be used for the Simulation of the Laser Alignment System
 *  an empty MCHepEvent will be generated (needed by OscarProducer). The actual simulation of 
 *  the laser beams is done in the SimWatcher attached to OscarProducer
 *
 *  $Date: 2011/09/16 06:23:27 $
 *  $Revision: 1.6 $
 *  \author Maarten Thomas
 */
// system include files
#include "FWCore/Framework/interface/Event.h"

// user include files
#include "Alignment/LaserAlignmentSimulation/plugins/LaserAlignmentProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h" 

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//
// constructors and destructor
//
LaserAlignmentProducer::LaserAlignmentProducer(const edm::ParameterSet&) :
  EDProducer(),
  theEvent(0)
{
  //register your products
  produces<edm::HepMCProduct>("unsmeared");

  //now do what ever other initialization is needed
}


LaserAlignmentProducer::~LaserAlignmentProducer()
{
  // no need to cleanup theEvent since it's done in HepMCProduct
}

// ------------ method called to produce the event  ------------
void LaserAlignmentProducer::produce(edm::Event& iEvent, const edm::EventSetup&)
{
  // create the event
  theEvent = new HepMC::GenEvent();

  // create a primary vertex
  HepMC::GenVertex * theVtx = new HepMC::GenVertex(HepMC::FourVector(0.,0.,0.));

  // add a particle to the vertex; this is needed to avoid crashes in OscarProducer. Use a 
  // electron neutrino, with zero energy and mass
  HepMC::GenParticle * theParticle = new HepMC::GenParticle(HepMC::FourVector(0.,0.,0.,0.),12,1);
  
  theVtx->add_particle_out(theParticle);

  // add the vertex to the event
  theEvent->add_vertex(theVtx);

  // set the event number
  theEvent->set_event_number(iEvent.id().event());
  // set the signal process id
  theEvent->set_signal_process_id(20);

  // create an empty output collection
  std::auto_ptr<edm::HepMCProduct> theOutput(new edm::HepMCProduct());
  theOutput->addHepMCData(theEvent);
   
  // put the output to the event
  iEvent.put(theOutput);
}

//define this as a plug-in

DEFINE_FWK_MODULE(LaserAlignmentProducer);
