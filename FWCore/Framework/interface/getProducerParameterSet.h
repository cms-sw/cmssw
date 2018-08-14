#ifndef FWCore_Framework_getProducerParameterSet_h
#define FWCore_Framework_getProducerParameterSet_h
// -*- C++ -*-
//
// Package:     Framework
// Function:  getProducerParameterSet
//
/**\function edm::getProducerParameterSet

 Description: Returns the ParameterSet of the module that produced
              the product corresponding to the Provenance.

              This shouldn't ever fail if the Provenance and
              other objects it uses have been properly initialized,
              which should always be the case when called from a
              module using the Provenance from a Handle returned
              by a function like getByToken. If it does fail it will
              throw an exception.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  7 September 2017

namespace edm {

  class ParameterSet;
  class Provenance;

  ParameterSet const*
  getProducerParameterSet(Provenance const& provenance);
}
#endif
