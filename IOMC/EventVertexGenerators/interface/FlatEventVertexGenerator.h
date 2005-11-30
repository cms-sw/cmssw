#ifndef IOMC_FlatEventVertexGenerator_H
#define IOMC_FlatEventVertexGenerator_H

#include "IOMC/EventVertexGenerators/interface/BaseEventVertexGenerator.h"

/**
 * Generate event vertices according to a flat distribution. 
 * Attention: All values are assumed to be mm!
 * \author Stephan Wynhoff
 */
class FlatEventVertexGenerator : public BaseEventVertexGenerator
{
public:
  FlatEventVertexGenerator(const edm::ParameterSet & p);
  virtual ~FlatEventVertexGenerator();

  /// return a new event vertex
  virtual Hep3Vector * newVertex();

  /** return the last generated event vertex.
   *  If no vertex has been generated yet, a NULL pointer is returned. */
  virtual Hep3Vector * lastVertex();

  /// set minimum in X in mm
  void minX(double min=0);
  /// set minimum in Y in mm
  void minY(double min=0);
  /// set minimum in Z in mm
  void minZ(double min=0);

  /// set maximum in X in mm
  void maxX(double max=0);
  /// set maximum in Y in mm
  void maxY(double max=0);
  /// set maximum in Z in mm
  void maxZ(double max=0);
  
private:
  /** Copy constructor */
  FlatEventVertexGenerator(const FlatEventVertexGenerator &p);
  /** Copy assignment operator */
  FlatEventVertexGenerator&  operator = (const FlatEventVertexGenerator & rhs );
private:
  edm::ParameterSet m_pFlatEventVertexGenerator;
  double myMinX, myMinY, myMinZ;
  double myMaxX, myMaxY, myMaxZ;
  Hep3Vector * myVertex;
};

#endif
