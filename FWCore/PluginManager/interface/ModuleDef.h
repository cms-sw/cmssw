#ifndef FWCore_PluginManager_ModuleDef_h
#define FWCore_PluginManager_ModuleDef_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     ModuleDef
// 
/**\class ModuleDef ModuleDef.h FWCore/PluginManager/interface/ModuleDef.h

 Description: Provided to easy migration from Seal PluginManager

 Usage:
    Use only temporarily to make it easier to migrate from Seal PluginManager
*/
//
// Original Author:  Chris Jones
//         Created:  Sat Apr  7 16:18:51 EDT 2007
// $Id: ModuleDef.h,v 1.3 2007/04/13 10:39:42 wmtan Exp $
//

#if !defined(DEFINE_SEAL_MODULE)
#define DEFINE_SEAL_MODULE() typedef int pluginManagerNeedsSemiColon
#endif

#endif
