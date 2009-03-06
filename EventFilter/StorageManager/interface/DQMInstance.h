#ifndef _DQMInstance_h
#define _DQMInstance_h

/*
   Author: William Badgett, FNAL

   Description:
     Container class for one snapshot instance of a collection of 
     collated DQM objects

   $Id: DQMInstance.h,v 1.7.4.1 2008/10/29 19:24:19 biery Exp $
*/

#include <string>
#include <vector>
#include <map>

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageService/interface/MessageServicePresence.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "PluginManager/PluginManager.h"

#include "TFile.h"
#include "TTimeStamp.h"
#include "TObject.h"

namespace stor 
{
  class DQMFolder
  {
    public:
      DQMFolder();
     ~DQMFolder();
      std::map<std::string, TObject *> dqmObjects_;
  }; 

  class DQMGroup
  {
    public:
      DQMGroup(int readyTime);
     ~DQMGroup();
      std::map<std::string, DQMFolder *> dqmFolders_;
      int getNUpdates()             { return(nUpdates_);}
      int getReadyTime()            { return(readyTime_);}
      int getLastEvent()            { return(lastEvent_);}
      void setLastEvent(int lastEvent) { lastEvent_=lastEvent;}
      TTimeStamp * getFirstUpdate() { return(firstUpdate_);}
      TTimeStamp * getLastUpdate()  { return(lastUpdate_);}
      TTimeStamp * getLastServed()  { return(lastServed_);}
      bool isReady(int currentTime);
      bool wasServedSinceUpdate()   { return(wasServedSinceUpdate_);}
      void setServedSinceUpdate()   { wasServedSinceUpdate_=true;}
      void incrementUpdates();
      void setLastServed()          { lastServed_->Set();}

    protected:
      TTimeStamp            *firstUpdate_;
      TTimeStamp            *lastUpdate_;
      TTimeStamp            *lastServed_;
      int                    nUpdates_;
      int                    readyTime_;
      int                    lastEvent_;
      bool                   wasServedSinceUpdate_;
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
      int getLastEvent()            { return(lastEvent_);}
      int getLumiSection()          { return(lumiSection_);}
      int getInstance()             { return(instance_);}
      int getPurgeTime()            { return(purgeTime_);}
      int getReadyTime()            { return(readyTime_);}

      TTimeStamp * getFirstUpdate() { return(firstUpdate_);}
      TTimeStamp * getLastUpdate()  { return(lastUpdate_);}
      int updateObject(std::string groupName,
		       std::string objectDirectory,
		       TObject   * object,
		       int         eventNumber);
      int writeFile(std::string filePrefix, bool endRunFlag);
      DQMGroup * getDQMGroup(std::string groupName);
      bool isReady(int currentTime);
      bool isStale(int currentTime);
      std::map<std::string, DQMGroup *> dqmGroups_;

      static std::string getSafeMEName(TObject *object);

    protected:  
      int                    runNumber_;
      int                    lastEvent_;
      int                    lumiSection_;
      int                    instance_;
      TTimeStamp            *firstUpdate_;
      TTimeStamp            *lastUpdate_;
      int                    nUpdates_;
      int                    purgeTime_;
      int                    readyTime_;
  }; 

  class DQMGroupDescriptor
  {
    public:
      DQMGroupDescriptor(DQMInstance *instance,DQMGroup *group);
     ~DQMGroupDescriptor();
      DQMInstance *instance_;
      DQMGroup    *group_;
  };
}


#endif
