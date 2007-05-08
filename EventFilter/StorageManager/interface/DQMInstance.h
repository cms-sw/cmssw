#ifndef _DQMInstance_h
#define _DQMInstance_h

/*
   Author: William Badgett, FNAL

   Description:
     Container class for one snapshot instance of a collection of 
     collated DQM objects

   $Id$
*/

#include <string>
#include <vector>
#include <map>

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageService/interface/MessageServicePresence.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PluginManager/PluginManager.h"

#include "TFile.h"
#include "TTimeStamp.h"
#include "TObject.h"

namespace stor 
{
  class DQMGroup
  {
    public:
      DQMGroup();
     ~DQMGroup();
      std::map<std::string, TObject *> dqmObjects_;
  }; 

  class DQMInstance
  {
    public:
      DQMInstance(int runNumber, 
		  int lumiSection, 
		  int instance,
		  int purgeTime,
		  int readyTime);

     ~DQMInstance();

      int getRunNumber()            { return(runNumber_);}
      int getLumiSection()          { return(lumiSection_);}
      int getInstance()             { return(instance_);}
      int getNUpdates()             { return(nUpdates_);}
      int getPurgeTime()            { return(purgeTime_);}
      int getReadyTime()            { return(readyTime_);}
      TTimeStamp * getFirstUpdate() { return(firstUpdate_);}
      TTimeStamp * getLastUpdate()  { return(lastUpdate_);}
      int updateObject(std::string groupName,
		       std::string objectDirectory,
		       TObject   * object);
      int writeFile(std::string filePrefix);
      bool isStale(int currentTime);
      bool isReady(int currentTime);
      DQMGroup * getDQMGroup(std::string groupName);

    protected:  
      int                    runNumber_;
      int                    lumiSection_;
      int                    instance_;
      TTimeStamp            *firstUpdate_;
      TTimeStamp            *lastUpdate_;
      int                    nUpdates_;
      int                    purgeTime_;
      int                    readyTime_;
      std::map<std::string, DQMGroup *> dqmGroups_;
  }; 
}


#endif
