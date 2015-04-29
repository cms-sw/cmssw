//FAMOS Headers
#include "FastSimulation/Event/interface/FSimEvent.h"

//C++ Headers

FSimEvent::FSimEvent(const edm::ParameterSet& kine) 
    : FBaseSimEvent(kine), id_(edm::EventID(0,0,0)), weight_(0)
{}
  
FSimEvent::~FSimEvent()
{}

void 
FSimEvent::fill(const HepMC::GenEvent& hev, edm::EventID& Id) {
  FBaseSimEvent::fill(hev);
  id_ = Id;
}

void
FSimEvent::fill(const std::vector<SimTrack>& simTracks, 
		const std::vector<SimVertex>& simVertices) {
  FBaseSimEvent::fill(simTracks,simVertices);
  id_ = edm::EventID();
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
FSimEvent::load(edm::SimTrackContainer & c, edm::SimTrackContainer & m) const
{
  for (unsigned int i=0; i<nTracks(); ++i) {
    //    SimTrack t = SimTrack(ip,p,iv,ig);
    const SimTrack& t = embdTrack(i);
    // Save all tracks
    c.push_back(t);
    // Save also some muons for later parameterization
    if ( abs(t.type()) == 13 && 
	 t.momentum().perp2() > 1.0 &&
	 fabs(t.momentum().eta()) < 3.0 &&
	 track(i).noEndVertex() ) {
      // Actually save the muon mother (and the attached muon) in case
      if ( !track(i).noMother() && track(i).mother().closestDaughterId() == (int)i ) {
	const SimTrack& T = embdTrack(track(i).mother().id());
	m.push_back(T);
      } 
      m.push_back(t);
    }
  }
}

void 
FSimEvent::load(edm::SimVertexContainer & c) const
{
  for (unsigned int i=0; i<nVertices(); ++i) {
    //    SimTrack t = SimTrack(ip,p,iv,ig);
    c.push_back(embdVertex(i));
  }
}


void 
FSimEvent::load(FSimVertexTypeCollection & c) const
{

  for (unsigned int i=0; i<nVertices(); ++i) {
    c.push_back(embdVertexType(i));
  }
}







