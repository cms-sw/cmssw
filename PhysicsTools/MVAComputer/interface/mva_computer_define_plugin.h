#ifndef PhysicsTools_MVAComputer_mva_computer_define_plugin_h
#define PhysicsTools_MVAComputer_mva_computer_define_plugin_h
// -*- C++ -*-
//
// Package:     MVAComputer
// Class  :     mva_computer_define_plugin
// 
/**\class mva_computer_define_plugin mva_computer_define_plugin.h PhysicsTools/MVAComputer/interface/mva_computer_define_plugin.h

 Description: A macro to instantiate a MVA Computer

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Jan 22 10:03:47 CST 2013
// $Id: mva_computer_define_plugin.h,v 1.1 2013/01/22 16:46:07 chrjones Exp $
//

// system include files

// user include files
#include "PhysicsTools/MVAComputer/interface/VarProcessor.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"


#define MVA_COMPUTER_DEFINE_PLUGIN(T) \
	DEFINE_EDM_PLUGIN(edmplugin::PluginFactory<PhysicsTools::VarProcessor::PluginFunctionPrototype>, \
	                  PhysicsTools::VarProcessor::Dummy, \
	                  "VarProcessor/" #T)

#endif
