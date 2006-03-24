#ifndef FWCore_Utilities_RandomNumberGenerator_h
#define FWCore_Utilities_RandomNumberGenerator_h
// -*- C++ -*-
//
// Package:     Utilities
// Class  :     RandomNumberGenerator
// 
/**\class RandomNumberGenerator RandomNumberGenerator.h FWCore/Utilities/interface/RandomNumberGenerator.h

 Description: Interface for obtaining random numbers

 Usage:
    This class is the abstract interface to a Service which provides access to random number generation. One accesses
the service using the Service system

   edm::Service<RandomNumberGenerator> rng;
   uint32_t seed = rng->mySeed();

The RandomNumberGenerator automatically knows what module is requesting the seed and will return the proper one for that
module.

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Mar  7 09:30:28 EST 2006
// $Id$
//

// system include files
#include "boost/cstdint.hpp"

// user include files

// forward declarations
namespace edm {
class RandomNumberGenerator
{

   public:
      RandomNumberGenerator() {}
      virtual ~RandomNumberGenerator() {}

      // ---------- const member functions ---------------------
      ///returns the random number seed appropriate to the module requesting the data. Will throw if no seed available.
      virtual uint32_t mySeed() const = 0;
   
   private:
      RandomNumberGenerator(const RandomNumberGenerator&); // stop default
      const RandomNumberGenerator& operator=(const RandomNumberGenerator&); // stop default
};

}
#endif
