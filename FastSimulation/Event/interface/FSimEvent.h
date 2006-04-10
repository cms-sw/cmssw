#ifndef FSIMEVENT_H
#define FSIMEVENT_H

#include "FastSimulation/Event/interface/FBaseSimEvent.h"

#include "SimDataFormats/Track/interface/EmbdSimTrack.h"
#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"
#include "DataFormats/Common/interface/EventID.h"
 
#include "CLHEP/Vector/LorentzVector.h"
 
#include <vector>
 
/** The FAMOS SimEvent: inherits from TSimEvent and FBaseSimEvent,
 *  where the latter provides FAMOS-specific event features (splitting
 *  proposed by Maya STAVRIANAKOU)
 *
 * An FSimEvent contains, at filling time, only particles from the 
 * RawHepEvent it is being filled with. Material Effects then 
 * update its content, so that it resembles the output of Geant
 * at the end of the material effect processing.
 *
 * Important : As in TSimEvent, all distances are in cm 
 *
 * \author Patrick Janot, CERN
 * \date: 9-Dec-2003
 *
 */

class FSimEvent : public FBaseSimEvent {

public:

  /// Default constructor
  FSimEvent();

  ///  usual virtual destructor
  virtual ~FSimEvent();

  /// fill the FBaseSimEvent from the current HepMC::GenEvent
  void fill(const HepMC::GenEvent & hev, edm::EventID & Id);

  ///Method to return the EventId
  virtual edm::EventID id() const;

  ///Method to return the event weight
  virtual float weight() const;
    
  /// Number of tracks
  virtual unsigned int nTracks() const;
  /// Number of vertices
  virtual unsigned int nVertices() const;
  /// Number of MC particles
  virtual unsigned int nGenParts() const;  

  /// return track container
  edm::EmbdSimTrackContainer* tracks() const;
  /// return vertex container
  edm::EmbdSimVertexContainer* vertices() const;
  /// return MC track container
  std::vector<HepMC::GenParticle*>* genparts() const;

private:

  edm::EventID id_;
  double weight_;

};

#endif // FSIMEVENT_H
