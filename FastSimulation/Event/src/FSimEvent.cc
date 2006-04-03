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

static  const EmbdSimVertex zeroVertex;
const EmbdSimVertex & 
FSimEvent::embdVertex(int i) const { 
  if (i>=0 && i<=(int)mySimVertices->size()) 
    return (*mySimVertices)[i]; 
  else 
    return zeroVertex;
}

static  const EmbdSimTrack zeroTrack;
const EmbdSimTrack & 
FSimEvent::embdTrack(int i) const { 
  if (i>=0 && i<=(int)mySimTracks->size()) 
    return (*mySimTracks)[i]; 
  else 
    return zeroTrack;
}

const HepMC::GenParticle* 
FSimEvent::embdGenpart(int i) const { 
  if (i>=0 && i<=(int)myGenParticles->size()) 
    return (*myGenParticles)[i]; 
  else 
    return 0;
}

// dummy for now
void FSimEvent::load(EmbdSimTrack & trk, int i) const {}
void FSimEvent::load(EmbdSimVertex & vtx, int i) const {}
void FSimEvent::load(HepMC::GenParticle & part, int i) const {}









