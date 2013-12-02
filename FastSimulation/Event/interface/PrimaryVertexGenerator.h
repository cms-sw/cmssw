#ifndef FastSimulation_Event_PrimaryVertexGenerator_H
#define FastSimulation_Event_PrimaryVertexGenerator_H

// Data Format Headers
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "TMatrixD.h"

class RandomEngineAndDistribution;

/** A class that generates a primary vertex for the event, in cm*/ 

class PrimaryVertexGenerator : public math::XYZVector {

public:
  /// Default constructor
  PrimaryVertexGenerator();
  
  /// Destructor
  virtual ~PrimaryVertexGenerator();

  /// Generation process (to be implemented)
  virtual void generate(RandomEngineAndDistribution const*) = 0;

  TMatrixD* boost() const;

  /// Return x0, y0, z0
  inline const math::XYZPoint& beamSpot() const { return beamSpot_; }

 protected:

  void setBoost(TMatrixD*);

  TMatrixD* boost_;
  math::XYZPoint beamSpot_;

};

#endif // PrimaryVertexGenerator_H
