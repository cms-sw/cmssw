#ifndef Fireworks_Core_FWDetailViewFactory_h
#define Fireworks_Core_FWDetailViewFactory_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewFactory
//
/**\class FWDetailViewFactory FWDetailViewFactory.h Fireworks/Core/interface/FWDetailViewFactory.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Jan 12 09:48:04 EST 2009
// $Id: FWDetailViewFactory.h,v 1.4 2009/10/31 22:37:35 chrjones Exp $
//

// system include files
#include "FWCore/PluginManager/interface/PluginFactory.h"

// user include files

// forward declarations

class FWDetailViewBase;

typedef edmplugin::PluginFactory<FWDetailViewBase*()> FWDetailViewFactory;

#define REGISTER_FWDETAILVIEW(_classname_,_name_) \
   DEFINE_EDM_PLUGIN(FWDetailViewFactory,_classname_,_classname_::classRegisterTypeName()+"@" # _name_ "@" # _classname_)


#endif
