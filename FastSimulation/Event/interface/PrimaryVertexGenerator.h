#ifndef FastSimulation_Event_PrimaryVertexGenerator_H
#define FastSimulation_Event_PrimaryVertexGenerator_H

// CLHEP Headers
#include "CLHEP/Vector/ThreeVector.h"

class RandomEngine;

/** A class that generates a primary vertex for the event, in cm*/ 

class PrimaryVertexGenerator : public Hep3Vector {

public:
  /// Default constructor
  PrimaryVertexGenerator();
  
  /// Destructor
  virtual ~PrimaryVertexGenerator() {;}

  /// Generation process (to be implemented)
  virtual void generate() = 0;

 protected:

  RandomEngine* random;

};

#endif // PrimaryVertexGenerator_H
