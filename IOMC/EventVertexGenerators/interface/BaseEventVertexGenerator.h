#ifndef IOMC_BaseEventVertexGenerator_H
#define IOMC_BaseEventVertexGenerator_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Random/JamesRandom.h"

/**
 * An abstract base class for classes creating event vertices. 
 * All values are assumed to be mm.
 * \author Stephan Wynhoff; adapted to CMSSW by Maya Stavrianakou
*/
class BaseEventVertexGenerator
{
public:
  /// constructor
  BaseEventVertexGenerator(const edm::ParameterSet & p, const long& seed);
  /// virtual destructor
  virtual ~BaseEventVertexGenerator();
  /// return a new event vertex
  virtual Hep3Vector * newVertex() = 0;
  /** return the last generated event vertex.
   *  If no vertex has been generated yet, a NULL pointer is returned. */
  virtual Hep3Vector * lastVertex() = 0;
private:
   edm::ParameterSet        m_pBaseEventVertexGenerator; 
protected:
   CLHEP::HepRandomEngine*  m_Engine ;
};

#endif
