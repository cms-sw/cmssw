#ifndef CommonAlignmentMonitor_AlignmentMonitorPluginFactory_h
#define CommonAlignmentMonitor_AlignmentMonitorPluginFactory_h
// -*- C++ -*-
//
// Package:     CommonAlignmentMonitor
// Class  :     AlignmentMonitorPluginFactory
// 
/**\class AlignmentMonitorPluginFactory AlignmentMonitorPluginFactory.h Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorPluginFactory.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Jim Pivarski
//         Created:  Mon Apr 23 15:29:01 CDT 2007
// $Id: AlignmentMonitorPluginFactory.h,v 1.1 2007/04/23 22:19:13 pivarski Exp $
//

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "Alignment/CommonAlignmentMonitor/interface/AlignmentMonitorBase.h"

// Forward declaration
namespace edm { class ParameterSet; }

typedef edmplugin::PluginFactory<AlignmentMonitorBase* (const edm::ParameterSet&) > AlignmentMonitorPluginFactory;

// // Forward declaration
// namespace edm { class ParameterSet; }
// 
// class AlignmentMonitorPluginFactory : 
//   public seal::PluginFactory<AlignmentMonitorBase* (const edm::ParameterSet&) >  
// {
//   
// public:
//   /// Constructor
//   AlignmentMonitorPluginFactory();
//   
//   /// Return the plugin factory (unique instance)
//   static AlignmentMonitorPluginFactory* get (void);
//   
//   /// Directly return the algorithm with given name and configuration
//   static AlignmentMonitorBase* getMonitor( std::string name, 
// 											   const edm::ParameterSet& config );
//   
// private:
//   static AlignmentMonitorPluginFactory theInstance;
//   
// };

#endif

