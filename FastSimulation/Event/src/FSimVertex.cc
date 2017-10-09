#include "FastSimulation/Event/interface/FSimVertex.h"

  /// Default constructor
FSimVertex::FSimVertex() : SimVertex(), mom_(nullptr), id_(-1) {;}
  
  /// constructor from the embedded vertex index in the FBaseSimEvent
FSimVertex::FSimVertex(const XYZTLorentzVector& v, int im, int id, FBaseSimEvent* mom) : 
  //    SimVertex(Hep3Vector(v.vect(),v.e(),im), mom_(mom), id_(id) 
  SimVertex(v.Vect(),v.T(),im,id), mom_(mom), id_(id),
  position_(v) {;}

std::ostream& operator <<(std::ostream& o , const FSimVertex& t) {
  return o << t;
}
