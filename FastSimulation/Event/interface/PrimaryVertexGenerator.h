#ifndef FastSimulation_Event_PrimaryVertexGenerator_H
#define FastSimulation_Event_PrimaryVertexGenerator_H

// Data Format Headers
#include "DataFormats/Math/interface/Vector3D.h"
#include "TMatrixD.h"

class RandomEngine;

/** A class that generates a primary vertex for the event, in cm*/ 

class PrimaryVertexGenerator : public math::XYZVector {

public:
  /// Default constructor
  PrimaryVertexGenerator();
  PrimaryVertexGenerator(const RandomEngine* engine);
  
  /// Destructor
  virtual ~PrimaryVertexGenerator();

  /// Generation process (to be implemented)
  virtual void generate() = 0;

  TMatrixD* boost() const;

 protected:

  void setBoost(TMatrixD*);

  const RandomEngine* random;
  TMatrixD* boost_;

};

#endif // PrimaryVertexGenerator_H
