#ifndef FBaseSimEvent_H
#define FBaseSimEvent_H

//#include "DataFormats/Common/interface/EventID.h"

#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/HepMC/GenVertex.h"
#include "CLHEP/HepMC/GenParticle.h"
#include "FastSimulation/Event/interface/KineParticleFilter.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

/** FSimEvent special features for FAMOS
 *
 * \author Patrick Janot, CERN
 * \date: 9-Dec-2003
 */

class FSimTrack;
class FSimVertex;
class HepPDTable;

class FBaseSimEvent : public HepMC::GenEvent {

public:

  /// Default constructor
  FBaseSimEvent();

  ///  usual virtual destructor
  ~FBaseSimEvent();

  /// fill the FBaseSimEvent from the current HepMC::GenEvent
  void fill(const HepMC::GenEvent& hev);
  
  /// print the original MCTruth event
  void printMCTruth(const HepMC::GenEvent& hev);

  /// Add the particles and their vertices to the list
  void addParticles(const HepMC::GenEvent& hev);

  /// print the FBaseSimEvent in an intelligible way
  void print() const;

  /// clear the FBaseSimEvent content before the next event
  void clear();

  /// Number of tracks
  unsigned int nTracks() const;
  /// Number of vertices
  unsigned int nVertices() const;
  /// Number of generator particles
  unsigned int nGenParts() const;

  /// Return track with given Id 
  const FSimTrack& track(int id) const;
  /// Return vertex with given Id 
  const FSimVertex& vertex(int id) const;
  /// return "reconstructed" charged tracks index.
  int chargedTrack(int id) const;

  /// Add a new track to the Event and to the various lists
  int addSimTrack(HepMC::GenParticle* part, 
		  HepMC::GenVertex* originVertex, 
		  int ig=-1);

  /// Add a new vertex to the Event and to the various lists
  int addSimVertex(HepMC::GenVertex* decayVertex,
		   HepMC::GenParticle* motherParticle=0,
		   int it=-1);


 protected:

  /// To have the same output as for OscarProducer (->FamosProducer)
  edm::EmbdSimTrackContainer* mySimTracks;
  edm::EmbdSimVertexContainer* mySimVertices;
  std::vector<HepMC::GenParticle*>* myGenParticles;

 private:

  std::vector<FSimTrack>* theSimTracks;
  std::vector<FSimVertex>* theSimVertices;

  /// Some internal array to work with.
  std::map<const HepMC::GenParticle*,HepMC::GenVertex*> myGenVertices;

  /// The particle filter
  KineParticleFilter myFilter;

  double sigmaVerteX;
  double sigmaVerteY;
  double sigmaVerteZ;

  HepPDTable * tab;

};

#endif // FBaseSimEvent_H
