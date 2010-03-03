// $Id: DQMInstance.h,v 1.13 2009/09/17 14:29:24 mommsen Exp $
/// @file: DQMInstance.h 

#ifndef StorageManager_DQMInstance_h
#define StorageManager_DQMInstance_h

#include <string>
#include <vector>
#include <map>

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageService/interface/MessageServicePresence.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TFile.h"
#include "TTimeStamp.h"
#include "TObject.h"

namespace stor 
{

  /**
   * A single DQM folder holding several histograms
   *
   * $Author: mommsen $
   * $Revision: 1.13 $
   * $Date: 2009/09/17 14:29:24 $
   */

  class DQMFolder
  {
    public:
      DQMFolder();
     ~DQMFolder();
      typedef std::map<std::string, TObject *> DQMObjectsMap;
      DQMObjectsMap dqmObjects_;
  }; 



  /**
   * A collection of DQM Folders under the same top-level name.
   *
   * $Author: mommsen $
   * $Revision: 1.13 $
   * $Date: 2009/09/17 14:29:24 $
   */

  class DQMGroup
  {
    public:
      DQMGroup(int readyTime, int expectedUpdates);
     ~DQMGroup();
      typedef std::map<std::string, DQMFolder *> DQMFoldersMap;
      DQMFoldersMap dqmFolders_;
      inline int getNUpdates() const             { return(nUpdates_);}
      inline int getReadyTime() const            { return(readyTime_);}
      inline int getLastEvent() const            { return(lastEvent_);}
      void setLastEvent(int lastEvent);
      inline TTimeStamp * getFirstUpdate() const { return(firstUpdate_);}
      inline TTimeStamp * getLastUpdate() const  { return(lastUpdate_);}
      inline TTimeStamp * getLastServed() const  { return(lastServed_);}
      bool isReady(int currentTime) const;
      inline bool isComplete() const;
      bool wasServedSinceUpdate()   { return(wasServedSinceUpdate_);}
      void setServedSinceUpdate();
      void setLastServed()          { lastServed_->Set();}

    protected:
      TTimeStamp            *firstUpdate_;
      TTimeStamp            *lastUpdate_;
      TTimeStamp            *lastServed_;
      int                    nUpdates_;
      int                    readyTime_;
      int                    expectedUpdates_;
      int                    lastEvent_;
      bool                   wasServedSinceUpdate_;
  }; 



  /**
   * Container class for one snapshot instance of a collection of 
   * collated DQM groups
   *
   * $Author: mommsen $
   * $Revision: 1.13 $
   * $Date: 2009/09/17 14:29:24 $
   */

  class DQMInstance
  {
    public:
      DQMInstance(int runNumber, 
		  int lumiSection, 
		  int instance,
		  int purgeTime,
                  int readyTime,
                  int expectedUpdates);

     ~DQMInstance();

      inline int getRunNumber() const   { return(runNumber_);}
      inline int getLastEvent() const   { return(lastEvent_);}
      inline int getLumiSection() const { return(lumiSection_);}
      inline int getInstance() const    { return(instance_);}
      inline int getPurgeTime() const   { return(purgeTime_);}
      inline int getReadyTime() const   { return(readyTime_);}

      inline TTimeStamp * getFirstUpdate() const { return(firstUpdate_);}
      inline TTimeStamp * getLastUpdate() const  { return(lastUpdate_);}
      int updateObject(std::string groupName,
		       std::string objectDirectory,
		       TObject   * object,
		       int         eventNumber);
      bool isReady(int currentTime) const;
      bool isStale(int currentTime) const;
      bool isComplete() const;

      double writeFile(std::string filePrefix, bool endRunFlag) const;
      DQMGroup * getDQMGroup(std::string groupName) const;
      typedef std::map<std::string, DQMGroup *> DQMGroupsMap;
      DQMGroupsMap dqmGroups_;

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
      int                    expectedUpdates_;
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


#endif // StorageManager_DQMInstance_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
