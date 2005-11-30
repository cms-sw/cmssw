#ifndef IOMC_GaussianEventVertexGenerator_H
#define IOMC_GaussianEventVertexGenerator_H

#include "IOMC/EventVertexGenerators/interface/BaseEventVertexGenerator.h"

/**
 * Generate event vertices according to a Gauss distribution. 
 * Attention: All values are assumed to be mm!
 * \author Stephan Wynhoff
 */
class GaussianEventVertexGenerator : public BaseEventVertexGenerator 
{
public:
  GaussianEventVertexGenerator(const edm::ParameterSet & p);
  virtual ~GaussianEventVertexGenerator();

  /// return a new event vertex
  virtual Hep3Vector * newVertex();

  /** return the last generated event vertex.
   *  If no vertex has been generated yet, a NULL pointer is returned. */
  virtual Hep3Vector * lastVertex();

  /// set resolution in X in mm
  void sigmaX(double s=1.0);
  /// set resolution in Y in mm
  void sigmaY(double s=1.0);
  /// set resolution in Z in mm
  void sigmaZ(double s=1.0);

  /// set mean in X in mm
  void meanX(double m=0) { myMeanX=m; };
  /// set mean in Y in mm
  void meanY(double m=0) { myMeanY=m; };
  /// set mean in Z in mm
  void meanZ(double m=0) { myMeanZ=m; };
  
private:
  /** Copy constructor */
  GaussianEventVertexGenerator(const GaussianEventVertexGenerator &p);
  /** Copy assignment operator */
  GaussianEventVertexGenerator&  operator = (const GaussianEventVertexGenerator & rhs );
private:
  edm::ParameterSet m_pGaussianEventVertexGenerator;
  double mySigmaX, mySigmaY, mySigmaZ;
  double myMeanX, myMeanY, myMeanZ;
  Hep3Vector * myVertex;
};

#endif
