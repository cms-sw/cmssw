#ifndef IOMC_BeamProfileVertexGenerator_H
#define IOMC_BeamProfileVertexGenerator_H

#include "IOMC/EventVertexGenerators/interface/BaseEventVertexGenerator.h"

/**
 * Generate event vertices according to a Gaussian distribution transverse
 * to beam direction (given by eta and phi
 * Attention: Units are assumed to be mm and radian!
 * \author Sunanda Banerjee
 */
class BeamProfileVertexGenerator : public BaseEventVertexGenerator
{
public:
  BeamProfileVertexGenerator(const edm::ParameterSet & p);
  virtual ~BeamProfileVertexGenerator();

  /// return a new event vertex
  virtual Hep3Vector * newVertex();

  /** return the last generated event vertex.
   *  If no vertex has been generated yet, a NULL pointer is returned. */
  virtual Hep3Vector * lastVertex();

  /// set resolution in X in mm
  void sigmaX(double s=1.0);
  /// set resolution in Y in mm
  void sigmaY(double s=1.0);

  /// set mean in X in mm
  void meanX(double m=0) {myMeanX=m;};
  /// set mean in Y in mm
  void meanY(double m=0) {myMeanY=m;};
  /// set mean in Z in mm
  void beamPos(double m=0) {myMeanZ=m;};

  /// set eta
  void eta(double m=0);
  /// set phi in radian
  void phi(double m=0)   {myPhi=m;};
  /// set type
  void setType(bool m=true) {myType=m;};
  
private:
  /** Copy constructor */
  BeamProfileVertexGenerator(const BeamProfileVertexGenerator &p);
  /** Copy assignment operator */
  BeamProfileVertexGenerator& operator = (const BeamProfileVertexGenerator& rhs);
private:
  edm::ParameterSet m_pBeamProfileVertexGenerator;
  double mySigmaX, mySigmaY;
  double myMeanX, myMeanY, myMeanZ;
  double myEta, myPhi, myTheta;
  bool   myType;
  Hep3Vector * myVertex;
};

#endif
