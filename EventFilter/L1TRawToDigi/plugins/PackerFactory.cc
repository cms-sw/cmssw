#include "PackerFactory.h"

#include "FWCore/Utilities/interface/EDMException.h"

EDM_REGISTER_PLUGINFACTORY(l1t::PackerFactoryT, "PackerFactory");

namespace l1t {
  const PackerFactory PackerFactory::instance_;

  std::shared_ptr<Packer> PackerFactory::make(const std::string& name) const {
    auto unpacker = std::shared_ptr<Packer>(PackerFactoryT::get()->create("l1t::" + name));

    if (unpacker.get() == nullptr) {
      throw edm::Exception(edm::errors::Configuration, "NoSourceModule") << "Cannot find a packer named " << name;
    }

    return unpacker;
  }
}  // namespace l1t
