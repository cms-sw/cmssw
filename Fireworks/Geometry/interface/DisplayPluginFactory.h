#ifndef Fireworks_Geometry_DisplayPluginFactory_h
#define Fireworks_Geometry_DisplayPluginFactory_h
// -*- C++ -*-
//
// Package:     Geometry
// Class  :     DisplayPluginFactory
// 
/**\class DisplayPluginFactory DisplayPluginFactory.h Fireworks/Geometry/interface/DisplayPluginFactory.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Mar 18 04:08:40 CDT 2010
// $Id: DisplayPluginFactory.h,v 1.1 2010/04/01 21:57:59 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Fireworks/Geometry/interface/DisplayPlugin.h"

// forward declarations
namespace fireworks {
  namespace geometry {
    typedef edmplugin::PluginFactory<DisplayPlugin*(void)> DisplayPluginFactory;
  }
}

#define DEFINE_FIREWORKS_GEOM_DISPLAY(type) \
static fireworks::geometry::DisplayPluginFactory::PMaker<type > EDM_PLUGIN_SYM(s_display , __LINE__ ) (#type)


#endif
