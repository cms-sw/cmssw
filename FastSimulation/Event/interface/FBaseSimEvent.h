#ifndef FastSimulation_Event_FBaseSimEvent_H
#define FastSimulation_Event_FBaseSimEvent_H

// Data Formats
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Math/interface/Point3D.h"

// HepPDT Headers
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// Famos Headers
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimDataFormats/NuclearInteractions/interface/FSimVertexType.h"
#include "FastSimDataFormats/NuclearInteractions/interface/FSimVertexTypeFwd.h"

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

namespace edm {
  class ParameterSet;
}

namespace HepMC {
  class GenEvent;
  class GenParticle;
  class GenVertex;
}

class FBaseSimEvent  
{

public:

  /// Default constructor
  FBaseSimEvent(const edm::ParameterSet& kine);

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

  /// fill the FBaseSimEvent from SimTrack's and SimVert'ices
  void fill(const std::vector<SimTrack>&, const std::vector<SimVertex>&);

  /// print the original MCTruth event
  void printMCTruth(const HepMC::GenEvent& hev);

  /// Add the particles and their vertices to the list
  void addParticles(const HepMC::GenEvent& hev);

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

  /// Return vertex with given Id 
  inline FSimVertexType& vertexType(int id) const;

  /// return "reconstructed" charged tracks index.
  int chargedTrack(int id) const;

  /// return embedded track with given id
  inline const SimTrack & embdTrack(int i) const;

  /// return embedded vertex with given id
  inline const SimVertex & embdVertex(int i) const;

  /// return embedded vertex type with given id
  inline const FSimVertexType & embdVertexType(int i) const;

  /// return MC track with a given id
  const HepMC::GenParticle* embdGenpart(int i) const;

  /// Add a new track to the Event and to the various lists
  int addSimTrack(const RawParticle* p, int iv, int ig=-1, 
		  const HepMC::GenVertex* ev=0);

  /// Add a new vertex to the Event and to the various lists
  int addSimVertex(const XYZTLorentzVector& decayVertex, int im=-1,
		   FSimVertexType::VertexType type = FSimVertexType::ANY);

  const KineParticleFilter& filter() const { return *myFilter; } 

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
  FSimVertexTypeCollection* theFSimVerticesType;
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

  double lateVertexPosition;

  //  Histos* myHistos;

};

#include "FastSimulation/Event/interface/FBaseSimEvent.icc"

#endif // FBaseSimEvent_H
