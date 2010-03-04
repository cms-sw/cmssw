// $Id: DQMInstance.h,v 1.14 2010/03/03 15:21:09 mommsen Exp $
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
   * $Revision: 1.14 $
   * $Date: 2010/03/03 15:21:09 $
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
   * $Revision: 1.14 $
   * $Date: 2010/03/03 15:21:09 $
   */

  class DQMGroup
  {
    public:
      DQMGroup(const int readyTime, const unsigned int expectedUpdates);
     ~DQMGroup();
      typedef std::map<std::string, DQMFolder *> DQMFoldersMap;
      DQMFoldersMap dqmFolders_;
      inline unsigned int getNUpdates() const    { return(nUpdates_);}
      inline int getReadyTime() const            { return(readyTime_);}
      inline int getLastEvent() const            { return(lastEvent_);}
      void setLastEvent(int lastEvent);
      inline TTimeStamp * getFirstUpdate() const { return(firstUpdate_);}
      inline TTimeStamp * getLastUpdate() const  { return(lastUpdate_);}
      inline TTimeStamp * getLastServed() const  { return(lastServed_);}
      bool isReady(int currentTime) const;
      bool isComplete() const;
      bool isStale(int currentTime) const;
      inline bool wasServedSinceUpdate() const   { return(wasServedSinceUpdate_);}
      void setServedSinceUpdate();
      inline void setLastServed() const          { lastServed_->Set();}

    protected:
      TTimeStamp            *firstUpdate_;
      TTimeStamp            *lastUpdate_;
      TTimeStamp            *lastServed_;
      unsigned int           nUpdates_;
      const int              readyTime_;
      const unsigned int     expectedUpdates_;
      int                    lastEvent_;
      bool                   wasServedSinceUpdate_;
  }; 



  /**
   * Container class for one snapshot instance of a collection of 
   * collated DQM groups
   *
   * $Author: mommsen $
   * $Revision: 1.14 $
   * $Date: 2010/03/03 15:21:09 $
   */

  class DQMInstance
  {
    public:
      DQMInstance(const int runNumber, 
		  const int lumiSection, 
		  const int instance,
		  const int purgeTime,
                  const int readyTime,
                  const unsigned int expectedUpdates);

     ~DQMInstance();

      inline int getRunNumber() const   { return(runNumber_);}
      inline int getLastEvent() const   { return(lastEvent_);}
      inline int getLumiSection() const { return(lumiSection_);}
      inline int getInstance() const    { return(instance_);}
      inline int getPurgeTime() const   { return(purgeTime_);}
      inline int getReadyTime() const   { return(readyTime_);}

      inline TTimeStamp * getFirstUpdate() const { return(firstUpdate_);}
      inline TTimeStamp * getLastUpdate() const  { return(lastUpdate_);}
      unsigned int updateObject(const std::string groupName,
		                const std::string objectDirectory,
		                TObject         * object,
		                const int         eventNumber);
      bool isReady(int currentTime) const;
      bool isStale(int currentTime) const;

      double writeFile(std::string filePrefix, bool endRunFlag) const;
      DQMGroup * getDQMGroup(std::string groupName) const;
      typedef std::map<std::string, DQMGroup *> DQMGroupsMap;
      DQMGroupsMap dqmGroups_;

      static std::string getSafeMEName(TObject *object);

    protected:  
      const int              runNumber_;
      int                    lastEvent_;
      const int              lumiSection_;
      const int              instance_;
      TTimeStamp            *firstUpdate_;
      TTimeStamp            *lastUpdate_;
      unsigned int           nUpdates_;
      const int              purgeTime_;
      const int              readyTime_;
      const unsigned int     expectedUpdates_;
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
