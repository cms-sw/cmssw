#ifndef FSimVertex_H
#define FSimVertex_H

#include<cmath>
#include <CLHEP/Vector/LorentzVector.h>
#include "CLHEP/HepMC/GenVertex.h"


class FBaseSimEvent;
class FSimTrack;

/** A class that mimics SimVertex, with enhanced features.
 *  Essentially an interface to FEmbdSimVertex.
 * \author Patrick Janot, CERN 
 * $Date: 9-Dec-2003
 */

class FSimVertex {

public:
  /// Default constructor
  FSimVertex() : me_(0), mom_(0) {;}
  
  /// constructor from the vertex index in the FBaseSimEvent
  FSimVertex(HepMC::GenVertex* ime, FBaseSimEvent* mom) : 
    me_(ime), mom_(mom) {;}

  /// retruns the vertex position
  HepLorentzVector position() const { 
    return me() ? me()->position() : HepLorentzVector(); }
  
  /// parent track
  const FSimTrack& parent() const; 

  /// first daughter
  const FSimTrack& daughter1() const;

  /// last daughter
  const FSimTrack& daughter2() const;

  /// no Parent track
  bool noParent() const { return  !me() || !me()->mother();}
  
  /// no Daughters
  bool  noDaughter() const { return !me() || me()->listChildren().size(); }

  /// the original GenVertex 
  HepMC::GenVertex* me() const { return me_; }

  /// the index in FBaseSimEvent
  int id() const { return me() ? -me()->barcode()+1 : -1; }

 private:

  HepMC::GenVertex* me_;
  const FBaseSimEvent* mom_;

  
};

#include<iosfwd>
std::ostream& operator <<(std::ostream& o , const FSimVertex& t);



#endif // FSimVertex_H
