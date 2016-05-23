#include <cmath>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

EDM_REGISTER_PLUGINFACTORY(l1t::UnpackerFactoryT,"UnpackerFactory");

namespace l1t {
   const UnpackerFactory UnpackerFactory::instance_;

   void
   getBXRange(int nbx, int& first, int& last)
   {
      last = std::floor(nbx / 2.);
      first = std::min(0, -last + (1 - nbx % 2));
   }

   std::shared_ptr<Unpacker>
   UnpackerFactory::make(const std::string& name) const
   {
      auto unpacker = std::shared_ptr<Unpacker>(UnpackerFactoryT::get()->create("l1t::" + name));

      if (unpacker.get() == 0) {
         throw edm::Exception(edm::errors::Configuration, "NoSourceModule")
            << "Cannot find an unpacker named " << name;
      }

      return unpacker;
   }
}
