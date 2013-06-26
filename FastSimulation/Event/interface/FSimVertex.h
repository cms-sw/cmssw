#ifndef FastSimulation_Event_FSimVertex_H
#define FastSimulation_Event_FSimVertex_H

// CMSSW Headers
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <vector>

class FBaseSimEvent;
class FSimTrack;

/** A class that mimics SimVertex, with enhanced features.
 *  Essentially an interface to SimVertex.
 * \author Patrick Janot, CERN 
 * $Date: 9-Dec-2003
 */

class FSimVertex : public SimVertex {

public:
  /// Default constructor
  FSimVertex();
  
  /// constructor from the embedded vertex index in the FBaseSimEvent
  FSimVertex(const math::XYZTLorentzVector& v, int im, int id, FBaseSimEvent* mom);

  /// parent track
  inline const FSimTrack& parent() const;

  /// The vector of daughter indices
  inline const std::vector<int>& daughters() const { return daugh_; }

  /// The number of daughters
  inline int nDaughters() const { return daugh_.size(); }

  /// ith daughter
  inline const FSimTrack& daughter(int i) const;

  /// no Daughters
  inline bool  noDaughter() const { return !nDaughters(); }

  /// the index in FBaseSimEvent
  inline int id() const { return id_; }

  inline void addDaughter(int i) { daugh_.push_back(i); }

  /// Temporary (until CMSSW moves to Mathcore) - No  ! Actually very useful
  inline const math::XYZTLorentzVector& position() const { return position_; }

  /// Reset the position (to be used with care)
  inline void setPosition(const math::XYZTLorentzVector& newPosition) {position_ = newPosition; }

  /// Simply returns the SimVertex
  inline const SimVertex& simVertex() const { return *this; }

 private:

  const FBaseSimEvent* mom_;
  int id_;    // The index in the FSimVertex vector
  std::vector<int> daugh_; // The indices of the daughters in FSimTrack

  math::XYZTLorentzVector position_;

};

#include<iosfwd>
std::ostream& operator <<(std::ostream& o , const FSimVertex& t);

#include "FastSimulation/Event/interface/FSimVertex.icc"

#endif // FSimVertex_H
