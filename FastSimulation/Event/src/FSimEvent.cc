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

void 
FSimEvent::fill(const HepMC::GenEvent& hev, edm::EventID& Id) { 
  FBaseSimEvent::fill(hev); 
  id_ = Id;
}
    
edm::EventID 
FSimEvent::id() const { 
  return id_; 
}
   
float FSimEvent::weight() const { 
  return weight_; 
}

edm::EmbdSimTrackContainer*
FSimEvent::tracks() const { return mySimTracks; }

edm::EmbdSimVertexContainer*
FSimEvent::vertices() const { return mySimVertices; }

std::vector<HepMC::GenParticle*>* 
FSimEvent::genparts() const { return myGenParticles; }

unsigned int 
FSimEvent::nTracks() const {
  return mySimTracks->size();
}

unsigned int 
FSimEvent::nVertices() const { 
  return mySimVertices->size();
}

unsigned int 
FSimEvent::nGenParts() const {
  return myGenParticles->size();
}










