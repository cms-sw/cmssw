#ifndef IOMC_BaseEventVertexGenerator_H
#define IOMC_BaseEventVertexGenerator_H

/**
 * An abstract base class for classes creating event vertices. 
 * All values are assumed to be mm.
 * \author Stephan Wynhoff; adapted to CMSSW by Maya Stavrianakou
*/
// $Id$

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace CLHEP {
  class HepRandomEngine;
  class Hep3Vector;
}

class BaseEventVertexGenerator
{
public:
  /// constructor
  explicit BaseEventVertexGenerator(const edm::ParameterSet & p);
  BaseEventVertexGenerator(const edm::ParameterSet & p, const long& seed);
  /// virtual destructor
  virtual ~BaseEventVertexGenerator();
  /// return a new event vertex
  virtual CLHEP::Hep3Vector * newVertex() = 0;
  /** return the last generated event vertex.
   *  If no vertex has been generated yet, a NULL pointer is returned. */
  virtual CLHEP::Hep3Vector * lastVertex() = 0;

protected:

  // Returns a reference to encourage users to use a reference
  // when initializing CLHEP distributions.  If a pointer
  // is used, then the distribution thinks it owns the engine
  // and will delete the engine when the distribution is destroyed
  // (a big problem since the distribution does not own the memory).
  CLHEP::HepRandomEngine& getEngine();

private:

  edm::ParameterSet        m_pBaseEventVertexGenerator; 

  CLHEP::HepRandomEngine*  m_engine ;
  bool m_ownsEngine;
};

#endif
