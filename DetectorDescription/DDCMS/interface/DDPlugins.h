#ifndef DETECTOR_DESCRIPTION_DD_PLUGINS_H
#define DETECTOR_DESCRIPTION_DD_PLUGINS_H

#include "DetectorDescription/DDCMS/interface/DDAlgoArguments.h"
#include "DetectorDescription/DDCMS/interface/DDUnits.h"
#include "DD4hep/Factories.h"
#include "DD4hep/Plugins.h"

namespace dd4hep {
  
  class SensitiveDetector;

  template <typename T> class DDCMSDetElementFactory : public dd4hep::PluginFactoryBase {
  public:
    static long create( dd4hep::Detector& detector,
			cms::DDParsingContext& context,
			dd4hep::xml::Handle_t element,
			dd4hep::SensitiveDetector& sensitive );
  };
}

namespace {
  template <typename P, typename S> class Factory;
  DD4HEP_PLUGIN_FACTORY_ARGS_4( long, dd4hep::Detector*,
				cms::DDParsingContext*, ns::xml_h*,
				dd4hep::SensitiveDetector* )
  {
    return dd4hep::DDCMSDetElementFactory<P>::create( *a0, *a1, *a2, *a3 );
  }
}

#define DECLARE_DDCMS_DETELEMENT(name,func)                             \
  DD4HEP_OPEN_PLUGIN(dd4hep,ddcms_det_element_##name) {                 \
    typedef DDCMSDetElementFactory< ddcms_det_element_##name > _IMP;    \
    template <> long                                                    \
      _IMP::create(dd4hep::Detector& d,                                 \
                   cms::DDParsingContext& c,                            \
                   xml::Handle_t e,                                     \
                   dd4hep::SensitiveDetector& h)                        \
    {  return func(d,c,e,h);       }                                    \
    DD4HEP_PLUGINSVC_FACTORY(ddcms_det_element_##name,name,             \
                             long(dd4hep::Detector*,cms::DDParsingContext*, \
                                  ns::xml_h*,dd4hep::SensitiveDetector*),__LINE__)  }

#endif
