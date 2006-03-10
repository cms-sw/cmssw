#include "SimDataFormats/Track/interface/EmbdSimTrack.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"

//FAMOS Headers
#include "FastSimulation/Event/interface/FSimEvent.h"

//C++ Headers

FSimEvent::FSimEvent() 
    : FBaseSimEvent(), id_(edm::EventID(0,0)), weight_(0)
{}
 
FSimEvent::~FSimEvent()
{}

void FSimEvent::fill(const HepMC::GenEvent& hev, edm::EventID& Id) { 
  FBaseSimEvent::fill(hev); 
  id_ = Id;
}
    
edm::EventID FSimEvent::id() const { 
  return id_; 
}
   
float FSimEvent::weight() const { 
  return weight_; 
}

unsigned int FSimEvent::nTracks() const {
  return 0;
}

unsigned int FSimEvent::nVertices() const { 
  return 0;
}

unsigned int FSimEvent::nGenParts() const {
  return 0;
}

// dummy for now
void FSimEvent::load(EmbdSimTrack & trk, int i) const {}
void FSimEvent::load(EmbdSimVertex & vtx, int i) const {}
void FSimEvent::load(HepMC::GenParticle & part, int i) const {}









