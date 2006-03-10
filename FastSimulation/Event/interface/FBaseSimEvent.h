#ifndef FBaseSimEvent_H
#define FBaseSimEvent_H

//#include "DataFormats/Common/interface/EventID.h"

#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/HepMC/GenVertex.h"
#include "CLHEP/HepMC/GenParticle.h"
#include "FastSimulation/Particle/interface/RawParticle.h"
#include "FastSimulation/Particle/interface/KineParticleFilter.h"
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

/** FSimEvent special features for FAMOS
 *
 * \author Patrick Janot, CERN
 * \date: 9-Dec-2003
 */

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

  /// return track container
  edm::EmbdSimTrackContainer* tracks() const;
  /// return vertex container
  edm::EmbdSimVertexContainer* vertices() const;
  /// return MC track container
  std::vector<HepMC::GenParticle*>* genparts() const;

  /// Number of tracks
  unsigned int nTracks() const;
  /// Number of vertices
  unsigned int nVertices() const;
  /// Number of MC particles
  unsigned int nGenParts() const;  

  /// return track with given id
  const EmbdSimTrack & embdTrack(int i) const;
  /// return vertex with given id
  const EmbdSimVertex & embdVertex(int i) const;
  /// return MC track with a given id
  const HepMC::GenParticle* embdGenpart(int i) const;

private:

  /// To have the same output as for OscarProducer (->FamosProducer)
  edm::EmbdSimTrackContainer* mySimTracks;
  edm::EmbdSimVertexContainer* mySimVertices;
  std::vector<HepMC::GenParticle*>* myGenParticles;

  /// Some internal array to work with.
  std::map<const HepMC::GenParticle*,HepMC::GenVertex*> myGenVertices;

  /// The particle filter
  KineParticleFilter myFilter;

  //  edm::EventID Id;
  std::map<int,std::string> particleNames;

  double sigmaVerteX;
  double sigmaVerteY;
  double sigmaVerteZ;

};

#endif // FBaseSimEvent_H
