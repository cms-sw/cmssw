#ifndef FastSimulation_Event_FSimVertex_H
#define FastSimulation_Event_FSimVertex_H

// CLHEP Headers
#include "CLHEP/Vector/LorentzVector.h"

// CMSSW Headers
#include "SimDataFormats/Vertex/interface/SimVertex.h"


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
  //  FSimVertex(int embd, FBaseSimEvent* mom);

  /// constructor from the embedded vertex index in the FBaseSimEvent
  FSimVertex(const HepLorentzVector& v, int im, int id, FBaseSimEvent* mom);

  /// retruns the vertex position (in cm !)
  inline HepLorentzVector position() const { return SimVertex::position()/10.; }
  
  /// parent track
  const FSimTrack& parent() const; 

  /// The vector of daughter indices
  inline const std::vector<int>& daughters() const { return daugh_; }

  /// The number of daughters
  inline int nDaughters() const { return daugh_.size(); }

  /// ith daughter
  const FSimTrack& daughter(int i) const;

  /// no Parent track
  //  inline bool noParent() const { return me().noParent();}
  
  /// no Daughters
  inline bool  noDaughter() const { return !nDaughters(); }

  /// The attached SimTrack
  //const SimVertex& me() const;

  /// the index in FBaseSimEvent
  inline int id() const { return id_; }

  inline void addDaughter(int i) { daugh_.push_back(i); }

 private:

  //  HepMC::GenVertex* me_;
  const FBaseSimEvent* mom_;
  //  int embd_; // The index in the SimVertex vector
  int id_;    // The index in the FSimVertex vector
  std::vector<int> daugh_; // The indices of the daughters in FSimTrack

  
};

#include<iosfwd>
std::ostream& operator <<(std::ostream& o , const FSimVertex& t);



#endif // FSimVertex_H
