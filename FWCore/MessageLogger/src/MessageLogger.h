#ifndef Services_MessageLogger_h
#define Services_MessageLogger_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageLogger
// 
/**\class MessageLogger MessageLogger.h FWCore/Services/interface/MessageLogger.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  W. Brown and M. FIschler 
//         Created:   Fri Nov 11 16:38:19 CST 2005
// $Id: MessageLogger.h,v 1.1 2005/11/14 16:36:49 fischler Exp $
//

// system include files

// user include files

// forward declarations

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"


namespace edm {
   namespace service {
      class MessageLogger {
public:
         MessageLogger(const ParameterSet&,ActivityRegistry&);
         
         void postBeginJob();
         void postEndJob();
         
         void preEventProcessing(const edm::EventID&, const edm::Timestamp&);
         void postEventProcessing(const Event&, const EventSetup&);
         
         void preModule(const ModuleDescription&);
         void postModule(const ModuleDescription&);
private:
         // put an ErrorLog object here, and maybe more
      };
   }
}



#endif
