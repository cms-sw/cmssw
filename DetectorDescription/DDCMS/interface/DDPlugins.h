#ifndef DETECTOR_DESCRIPTION_DD_PLUGINS_H
#define DETECTOR_DESCRIPTION_DD_PLUGINS_H

#include "DetectorDescription/DDCMS/interface/DDAlgoArguments.h"
#include "DD4hep/Factories.h"
#include "DD4hep/Plugins.h"

namespace dd4hep {

  template <typename T>
  class DDCMSDetElementFactory : public dd4hep::PluginFactoryBase {
  public:
    static long create(dd4hep::Detector& detector, cms::DDParsingContext& context, dd4hep::xml::Handle_t element);
  };
}  // namespace dd4hep

namespace {
  template <typename P, typename S>
  class Factory;
  DD4HEP_PLUGIN_FACTORY_ARGS_3(long, dd4hep::Detector*, cms::DDParsingContext*, ns::xml_h*) {
    return dd4hep::DDCMSDetElementFactory<P>::create(*a0, *a1, *a2);
  }
}  // namespace

#define DECLARE_DDCMS_DETELEMENT(name, func)                                                                   \
  DD4HEP_OPEN_PLUGIN(dd4hep, ddcms_det_element_##name) {                                                       \
    typedef DDCMSDetElementFactory<ddcms_det_element_##name> _IMP;                                             \
    template <>                                                                                                \
    long _IMP::create(dd4hep::Detector& d, cms::DDParsingContext& c, xml::Handle_t e) {                        \
      return func(d, c, e);                                                                                    \
    }                                                                                                          \
    DD4HEP_PLUGINSVC_FACTORY(                                                                                  \
        ddcms_det_element_##name, name, long(dd4hep::Detector*, cms::DDParsingContext*, ns::xml_h*), __LINE__) \
  }

#endif
