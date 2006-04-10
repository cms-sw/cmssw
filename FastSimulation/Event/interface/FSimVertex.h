#ifndef FSimVertex_H
#define FSimVertex_H

// CLHEP Headers
#include <CLHEP/Vector/LorentzVector.h>

// CMSSW Headers
#include "SimDataFormats/Vertex/interface/EmbdSimVertex.h"


class FBaseSimEvent;
class FSimTrack;

/** A class that mimics SimVertex, with enhanced features.
 *  Essentially an interface to EmbdSimVertex.
 * \author Patrick Janot, CERN 
 * $Date: 9-Dec-2003
 */

class FSimVertex {

public:
  /// Default constructor
  FSimVertex();
  
  /// constructor from the embedded vertex index in the FBaseSimEvent
  FSimVertex(int embd, FBaseSimEvent* mom);

  /// retruns the vertex position
  inline HepLorentzVector position() const { return me().position(); }
  
  /// parent track
  const FSimTrack& parent() const; 

  /// The vector of daughter indices
  inline const std::vector<int>& daughters() const { return daugh_; }

  /// The number of daughters
  inline int nDaughters() const { return daugh_.size(); }

  /// ith daughter
  const FSimTrack& daughter(int i) const;

  /// no Parent track
  inline bool noParent() const { return me().noParent();}
  
  /// no Daughters
  inline bool  noDaughter() const { return !nDaughters(); }

  /// The attached EmbdSimTrack
  const EmbdSimVertex& me() const;

  /// the index in FBaseSimEvent
  inline int id() const { return id_; }

  inline void addDaughter(int i) { daugh_.push_back(i); }

 private:

  //  HepMC::GenVertex* me_;
  const FBaseSimEvent* mom_;
  int embd_; // The index in the EmbdSimVertex vector
  int id_;    // The index in the FSimVertex vector
  std::vector<int> daugh_; // The indices of the daughters in FSimTrack

  
};

#include<iosfwd>
std::ostream& operator <<(std::ostream& o , const FSimVertex& t);



#endif // FSimVertex_H
