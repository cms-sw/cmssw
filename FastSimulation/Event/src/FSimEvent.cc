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

unsigned int 
FSimEvent::nTracks() const {
  return FBaseSimEvent::nTracks();
}

unsigned int 
FSimEvent::nVertices() const { 
  return FBaseSimEvent::nVertices();
}

unsigned int 
FSimEvent::nGenParts() const {
  return FBaseSimEvent::nGenParts();
}

void 
FSimEvent::load(edm::EmbdSimTrackContainer & c) const
{
  for (unsigned int i=0; i<nTracks(); ++i) {
    //    EmbdSimTrack t = EmbdSimTrack(ip,p,iv,ig);
    c.push_back(embdTrack(i));
  }
}

void 
FSimEvent::load(edm::EmbdSimVertexContainer & c) const
{
  for (unsigned int i=0; i<nVertices(); ++i) {
    //    EmbdSimTrack t = EmbdSimTrack(ip,p,iv,ig);
    c.push_back(embdVertex(i));
  }
}









