//==========================================================================
//  AIDA Detector description implementation 
//--------------------------------------------------------------------------
// Copyright (C) Organisation europeenne pour la Recherche nucleaire (CERN)
// All rights reserved.
//
// For the licensing terms see $DD4hepINSTALL/LICENSE.
// For the list of contributors see $DD4hepINSTALL/doc/CREDITS.
//
// Author     : M.Frank
//
//==========================================================================
//
// DDCMS is a detector description convention developed by the CMS experiment.
//
//==========================================================================
#ifndef DD4HEP_DDCMS_DDCMSPLUGINS_H
#define DD4HEP_DDCMS_DDCMSPLUGINS_H

// Framework includes
#include "DetectorDescription/DDCMS/interface/DDCMS.h"
#include "DD4hep/Plugins.h"
#include "CLHEP/Units/SystemOfUnits.h"

/// Namespace for the AIDA detector description toolkit
namespace dd4hep {

  // Forward declarations
  class SensitiveDetector;
  
  /// Standard factory to create Detector elements from an XML representation.
  /**
   *  \author  M.Frank
   *  \version 1.0
   *  \date    2012/07/31
   *  \ingroup DD4HEP_CMS
   */
  template <typename T> class DDCMSDetElementFactory : public PluginFactoryBase {
  public:
    static long create(Detector&            dsc,
                       cms::ParsingContext& ctx,
                       xml::Handle_t        elt,
                       SensitiveDetector&   sens);
  };
}     /* End namespace dd4hep          */

namespace {

  /// Forward declartion of the base factory template
  template <typename P, typename S> class Factory;
  DD4HEP_PLUGIN_FACTORY_ARGS_4(long,dd4hep::Detector*,dd4hep::cms::ParsingContext*,ns::xml_h*,dd4hep::SensitiveDetector*)
  {    return dd4hep::DDCMSDetElementFactory<P>::create(*a0,*a1,*a2,*a3);                     }
}

#define DECLARE_DDCMS_DETELEMENT(name,func)                             \
  DD4HEP_OPEN_PLUGIN(dd4hep,ddcms_det_element_##name) {                 \
    typedef DDCMSDetElementFactory< ddcms_det_element_##name > _IMP;    \
    template <> long                                                    \
      _IMP::create(dd4hep::Detector& d,                                 \
                   cms::ParsingContext& c,                              \
                   xml::Handle_t e,                                     \
                   SensitiveDetector& h)                                \
    {  return func(d,c,e,h);       }                                    \
    DD4HEP_PLUGINSVC_FACTORY(ddcms_det_element_##name,name,             \
                             long(dd4hep::Detector*,dd4hep::cms::ParsingContext*, \
                                  ns::xml_h*,dd4hep::SensitiveDetector*),__LINE__)  }

#endif /* DD4HEP_DDCMS_DDCMSPLUGINS_H  */
