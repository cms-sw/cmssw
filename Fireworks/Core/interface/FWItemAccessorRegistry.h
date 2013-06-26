#ifndef Fireworks_Core_FWItemAccessorRegistry_h
#define Fireworks_Core_FWItemAccessorRegistry_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemAccessorRegistry 
//
/**\class FWItemAccessorRegistry FWItemAccessorRegistry.h Fireworks/Core/src/FWItemAccessorRegistry.h

   Description: Registry for all th FWItemAccessorBase derived classes that can be loaded via the
                plugin manager. Those classes are to be used to have specialized versions of
                the accessors for objects that root does not consider as collections. 

   Usage:
    <usage>

 */
//
// Original Author:  Giulio Eulisse 
//         Created:  Thu Feb 18 00:00:00 EDT 2010
// $Id: FWItemAccessorRegistry.h,v 1.2 2010/03/01 09:43:01 eulisse Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/register_itemaccessorbase_macro.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

// forward declarations

class FWItemAccessorBase;
class TClass;

typedef FWItemAccessorBase* (IAccessorCreator)(const TClass *);
typedef edmplugin::PluginFactory<IAccessorCreator> FWItemAccessorRegistry;

#define REGISTER_FWITEMACCESSOR(_name_,_type_,_purpose_) \
   DEFINE_FWITEMACCESSOR_METHODS(_name_,_type_,_purpose_); \
   DEFINE_EDM_PLUGIN(FWItemAccessorRegistry,_name_,_name_::classRegisterTypeName()+"@"+_name_::classPurpose()+"@" # _name_)

#define REGISTER_TEMPLATE_FWITEMACCESSOR(_name_,_type_,_purpose_) \
   DEFINE_TEMPLATE_FWITEMACCESSOR_METHODS(_name_,_type_,_purpose_); \
   DEFINE_EDM_PLUGIN(FWItemAccessorRegistry,_name_,_name_::classRegisterTypeName()+"@"+_name_::classPurpose()+"@" # _name_)

#endif
