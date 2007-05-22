#ifndef FastSimulation_Event_FBaseSimEvent_H
#define FastSimulation_Event_FBaseSimEvent_H

//Framework Headers
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Data Formats
#include "DataFormats/Candidate/interface/CandidateFwd.h"

// HepPDT Headers
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// Famos Headers
#include "FastSimulation/Particle/interface/RawParticle.h"


#include <map>
#include <vector>

/** FSimEvent special features for FAMOS
 *
 * \author Patrick Janot, CERN
 * \date: 9-Dec-2003
 */

//class FSimEvent;
class FSimTrack;
class FSimVertex;
class KineParticleFilter;

class SimTrack;
class SimVertex;
class PrimaryVertexGenerator;
class RandomEngine;
//class Histos;

namespace HepMC {
  class GenEvent;
  class GenParticle;
}

class FBaseSimEvent  
{

public:

  /// Default constructor
  FBaseSimEvent(const edm::ParameterSet& kine);

  FBaseSimEvent(const edm::ParameterSet& vtx,
		const edm::ParameterSet& kine,
		const RandomEngine* engine);

  ///  usual virtual destructor
  ~FBaseSimEvent();

  /// Initialize the particle data table
  void initializePdt(const HepPDT::ParticleDataTable* aPdt);

  /// Get the pointer to the particle data table
  inline const HepPDT::ParticleDataTable* theTable() const { 
    return pdt;
  }

  /// fill the FBaseSimEvent from the current HepMC::GenEvent
  void fill(const HepMC::GenEvent& hev);

  /// fill the FBaseSimEvent from the current reco::CandidateCollection
  void fill(const reco::CandidateCollection& hev);

  /// fill the FBaseSimEvent from SimTrack's and SimVert'ices
  void fill(const std::vector<SimTrack>&, const std::vector<SimVertex>&);
  
  /// print the original MCTruth event
  void printMCTruth(const HepMC::GenEvent& hev);

  /// Add the particles and their vertices to the list
  void addParticles(const HepMC::GenEvent& hev);
  void addParticles(const reco::CandidateCollection& myGenParticles);

  /// print the FBaseSimEvent in an intelligible way
  void print() const;

  /// clear the FBaseSimEvent content before the next event
  void clear();


  /// Add an id in the vector of charged tracks id's
  void addChargedTrack(int id);

  /// Number of tracks
  inline unsigned int nTracks() const {
    return nSimTracks;
  }

  /// Number of vertices
  inline unsigned int nVertices() const { 
    return nSimVertices;
  }

  /// Number of generator particles
  inline unsigned int nGenParts() const {
    return nGenParticles;
  }

  /// Number of "reconstructed" charged tracks
  inline unsigned int nChargedTracks() const {
    return nChargedParticleTracks;
  }

  /// Return track with given Id 
  inline FSimTrack& track(int id) const;

  /// Return vertex with given Id 
  inline FSimVertex& vertex(int id) const;

  /// return "reconstructed" charged tracks index.
  int chargedTrack(int id) const;

  /// return embedded track with given id
  inline const SimTrack & embdTrack(int i) const;

  /// return embedded vertex with given id
  inline const SimVertex & embdVertex(int i) const;

  /// return MC track with a given id
  const HepMC::GenParticle* embdGenpart(int i) const;

  /// Add a new track to the Event and to the various lists
  int addSimTrack(const RawParticle* p, int iv, int ig=-1);

  /// Add a new vertex to the Event and to the various lists
  int addSimVertex(const XYZTLorentzVector& decayVertex,int im=-1);

  const KineParticleFilter& filter() const { return *myFilter; } 

  PrimaryVertexGenerator* thePrimaryVertexGenerator() const { return theVertexGenerator; }


 protected:

  /// The pointer to the vector of FSimTrack's 
  inline std::vector<FSimTrack>* tracks() const { 
    return theSimTracks; 
  }

  /// The pointer to the vector of FSimVertex's 
  inline std::vector<FSimVertex>* vertices() const { 
    return theSimVertices; 
  }

  /// The pointer to the vector of GenParticle's 
  inline std::vector<HepMC::GenParticle*>* genparts() const { 
    return theGenParticles; 
  }



 private:

  std::vector<FSimTrack>* theSimTracks;
  std::vector<FSimVertex>* theSimVertices;
  std::vector<HepMC::GenParticle*>* theGenParticles;

  std::vector<unsigned>* theChargedTracks;

  unsigned int nSimTracks;
  unsigned int nSimVertices;
  unsigned int nGenParticles;
  unsigned int nChargedParticleTracks;

  unsigned int theTrackSize;
  unsigned int theVertexSize;
  unsigned int theGenSize;
  unsigned int theChargedSize;
  unsigned int initialSize;

  /// The particle filter
  KineParticleFilter* myFilter;

  double sigmaVerteX;
  double sigmaVerteY;
  double sigmaVerteZ;

  const ParticleDataTable * pdt;

  PrimaryVertexGenerator* theVertexGenerator;

  const RandomEngine* random;

  //  Histos* myHistos;

};

#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"

static FSimTrack oTrack;
inline FSimTrack& FBaseSimEvent::track(int i) const { 
  return (i>=0 && i<(int)nTracks()) ? (*theSimTracks)[i] : oTrack; }

static FSimVertex oVertex;
inline FSimVertex& FBaseSimEvent::vertex(int i) const { 
  return (i>=0 && i<(int)nVertices()) ? (*theSimVertices)[i] : oVertex; }

inline const SimTrack& FBaseSimEvent::embdTrack(int i) const { 
  return (*theSimTracks)[i].simTrack(); }

inline const SimVertex& FBaseSimEvent::embdVertex(int i) const { 
  return (*theSimVertices)[i].simVertex(); }


#endif // FBaseSimEvent_H
