#ifndef FWCore_Framework_ScheduleInfo_h
#define FWCore_Framework_ScheduleInfo_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ScheduleInfo
// 
/**\class ScheduleInfo ScheduleInfo.h FWCore/Framework/interface/ScheduleInfo.h

 Description: Provides module and path information to EDLoopers

 Usage:
    This class allows EDLoopers to find out about the configuration of module and paths in the process.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jul 15 19:39:56 CEST 2010
// $Id: ScheduleInfo.h,v 1.1 2010/07/22 15:00:27 chrjones Exp $
//

// system include files
#include <vector>
#include <string>

// user include files

// forward declarations
namespace edm {
   class ParameterSet;
   class Schedule;
   
   class ScheduleInfo {
      
   public:
      ScheduleInfo(const Schedule*);
      virtual ~ScheduleInfo();
      
      // ---------- const member functions ---------------------
      
      ///adds to oLabelsToFill the labels for all modules used in the process
      void availableModuleLabels(std::vector<std::string>& oLabelsToFill) const;
      
      /**returns a pointer to the parameters for the module with label iLabel, returns 0 if 
       no module exists with that label.
       */
      const edm::ParameterSet* parametersForModule(const std::string& iLabel) const;
      
      ///adds to oLabelsToFill the labels for all paths in the process
      void availablePaths( std::vector<std::string>& oLabelsToFill) const;
      
      ///add to oLabelsToFill in execution order the labels of all modules in path iPathLabel
      void modulesInPath(const std::string& iPathLabel,
                         std::vector<std::string>& oLabelsToFill) const;
      // ---------- static member functions --------------------      
      
      // ---------- member functions ---------------------------
      
   private:
      //ScheduleInfo(const ScheduleInfo&); // stop default
      
      //const ScheduleInfo& operator=(const ScheduleInfo&); // stop default
      
      // ---------- member data --------------------------------
      const Schedule* schedule_;
      
   };

}

#endif
