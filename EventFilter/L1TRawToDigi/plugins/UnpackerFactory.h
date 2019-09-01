#ifndef EventFilter_L1TRawToDigi_UnpackerFactory_h
#define EventFilter_L1TRawToDigi_UnpackerFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "EventFilter/L1TRawToDigi/interface/Unpacker.h"

namespace l1t {
  typedef Unpacker*(unpack_fct)();
  typedef edmplugin::PluginFactory<unpack_fct> UnpackerFactoryT;

  class UnpackerFactory {
  public:
    inline static const UnpackerFactory* get() { return &instance_; };
    std::shared_ptr<Unpacker> make(const std::string&) const;

  private:
    UnpackerFactory(){};
    static const UnpackerFactory instance_;
  };
}  // namespace l1t

#define DEFINE_L1T_UNPACKER(type) DEFINE_EDM_PLUGIN(l1t::UnpackerFactoryT, type, #type)

#endif
