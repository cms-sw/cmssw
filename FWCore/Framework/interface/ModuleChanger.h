#ifndef FWCore_Framework_ModuleChanger_h
#define FWCore_Framework_ModuleChanger_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ModuleChanger
// 
/**\class ModuleChanger ModuleChanger.h FWCore/Framework/interface/ModuleChanger.h

 Description: Handles modifying a module after the job has started

 Usage:
    This class is used by an EDLooper at the end of a loop in order to modify the
 parameters of a module.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jul 15 15:05:17 EDT 2010
// $Id: ModuleChanger.h,v 1.1 2010/07/22 15:00:27 chrjones Exp $
//

// system include files
#include <string>

// user include files

// forward declarations

namespace edm {
   class ParameterSet;
   class Schedule;
   
   class ModuleChanger {

   public:
      ModuleChanger(Schedule*);
      virtual ~ModuleChanger();

      // ---------- const member functions ---------------------
      bool changeModule(const std::string& iLabel,
                        const ParameterSet& iPSet) const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      ModuleChanger(const ModuleChanger&); // stop default

      const ModuleChanger& operator=(const ModuleChanger&); // stop default

      // ---------- member data --------------------------------
      Schedule* schedule_;
   };
}
#endif
