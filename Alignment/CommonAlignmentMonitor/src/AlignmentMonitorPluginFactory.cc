// -*- C++ -*-
//
// Package:     CommonAlignmentMonitor
// Class  :     AlignmentMonitorPluginFactory
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Mon Apr 23 15:30:20 CDT 2007
// $Id: AlignmentMonitorPluginFactory.cc,v 1.2 2007/12/04 23:29:27 ratnik Exp $
//

// system include files

// user include files
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h"

EDM_REGISTER_PLUGINFACTORY(AlignmentMonitorPluginFactory, "AlignmentMonitorPluginFactory");

// //
// // constants, enums and typedefs
// //
// 
// //
// // static data member definitions
// //
// 
// AlignmentMonitorPluginFactory AlignmentMonitorPluginFactory::theInstance;
// 
// //
// // constructors and destructor
// //
// 
// //__________________________________________________________________________________________________
// AlignmentMonitorPluginFactory::AlignmentMonitorPluginFactory()
//    : seal::PluginFactory<AlignmentMonitorBase* (const edm::ParameterSet&)>("AlignmentMonitorPluginFactory")
// { }
// 
// //__________________________________________________________________________________________________
// 
// AlignmentMonitorPluginFactory* AlignmentMonitorPluginFactory::get() {
//   return &theInstance;
// }
// 
// //__________________________________________________________________________________________________
// 
// AlignmentMonitorBase* 
// AlignmentMonitorPluginFactory::getMonitor(std::string name, const edm::ParameterSet& config) {
//   return theInstance.create(name, config);
// }
