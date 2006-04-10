#ifndef FBaseSimEvent_H
#define FBaseSimEvent_H

//#include "DataFormats/Common/interface/EventID.h"

// CLHEP Headers
#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/HepMC/GenVertex.h"
#include "CLHEP/HepMC/GenParticle.h"

// CMSSW Headers
#include "SimDataFormats/Track/interface/EmbdSimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertexContainer.h"

// FAMOS Headers
#include "FastSimulation/Event/interface/KineParticleFilter.h"

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
  FSimTrack& track(int id) const;
  /// Return vertex with given Id 
  FSimVertex& vertex(int id) const;
  /// return "reconstructed" charged tracks index.
  int chargedTrack(int id) const;

  /// return embedded track with given id
  const EmbdSimTrack & embdTrack(int i) const;
  /// return embedded vertex with given id
  const EmbdSimVertex & embdVertex(int i) const;
  /// return MC track with a given id
  const HepMC::GenParticle* embdGenpart(int i) const;

  /// Add a new track to the Event and to the various lists
  //  int addSimTrack(HepMC::GenParticle* part, 
  //		  HepMC::GenVertex* originVertex, 
  //		  int ig=-1);
  int addSimTrack(const RawParticle* p, int iv, int ig=-1);

  /// Add a new vertex to the Event and to the various lists
  //  int addSimVertex(HepMC::GenVertex* decayVertex,int im=-1);
  int addSimVertex(const CLHEP::HepLorentzVector& decayVertex,int im=-1);

  const KineParticleFilter filter() const { return myFilter; } 


 protected:

  /// To have the same output as for OscarProducer (->FamosProducer)
  edm::EmbdSimTrackContainer* mySimTracks;
  edm::EmbdSimVertexContainer* mySimVertices;
  std::vector<HepMC::GenParticle*>* myGenParticles;

 private:

  std::vector<FSimTrack>* theSimTracks;
  std::vector<FSimVertex>* theSimVertices;

  /// Some internal array to work with.
  std::map<const HepMC::GenParticle*,int> myGenVertices;

  /// The particle filter
  KineParticleFilter myFilter;

  double sigmaVerteX;
  double sigmaVerteY;
  double sigmaVerteZ;

  HepPDTable * tab;

};

#endif // FBaseSimEvent_H
