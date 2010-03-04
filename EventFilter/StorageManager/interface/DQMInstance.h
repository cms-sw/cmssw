// $Id: DQMInstance.h,v 1.15 2010/03/04 11:19:40 mommsen Exp $
/// @file: DQMInstance.h 

#ifndef StorageManager_DQMInstance_h
#define StorageManager_DQMInstance_h

#include <string>
#include <vector>
#include <map>

#include "IOPool/Streamer/interface/DQMEventMessage.h"

#include "TFile.h"
#include "TObject.h"
#include "TTimeStamp.h"

namespace stor 
{
  
  /**
   * A single DQM folder holding several histograms
   *
   * $Author: mommsen $
   * $Revision: 1.15 $
   * $Date: 2010/03/04 11:19:40 $
   */

  class DQMFolder
  {
  public:
    DQMFolder();
    ~DQMFolder();
    void addObjects(std::vector<TObject *>);
    void fillObjectVector(std::vector<TObject*>&) const;
    unsigned int writeObjects() const;

    static std::string getSafeMEName(TObject *object);

  public: // old SM Proxy code relies on public access
    typedef std::map<std::string, TObject *> DQMObjectsMap;
    DQMObjectsMap dqmObjects_;
  }; 



  /**
   * A collection of DQM Folders under the same top-level name.
   *
   * $Author: mommsen $
   * $Revision: 1.15 $
   * $Date: 2010/03/04 11:19:40 $
   */

  class DQMGroup
  {
  public:
    DQMGroup(const time_t readyTime, const unsigned int expectedUpdates);
    ~DQMGroup();

    void addEvent(std::auto_ptr<DQMEvent::TObjectTable>);
    size_t populateTable(DQMEvent::TObjectTable&) const;
    unsigned int fillFile(TFile*, const TString& runString) const;

    inline unsigned int getNUpdates() const    { return(nUpdates_);}
    inline TTimeStamp * getFirstUpdate() const { return(firstUpdate_);}
    inline TTimeStamp * getLastUpdate() const  { return(lastUpdate_);}
    inline TTimeStamp * getLastServed() const  { return(lastServed_);}
    inline bool wasServedSinceUpdate() const   { return(wasServedSinceUpdate_);}
    inline void setServedSinceUpdate()         { wasServedSinceUpdate_ = true; }
    inline bool isReady(time_t now) const      { return ( isComplete() || isStale(now) ); }
    inline bool isComplete() const             { return ( expectedUpdates_ == nUpdates_ ); }
    bool isStale(time_t now) const;

  public: // old SM Proxy code relies on public access
    typedef std::map<std::string, DQMFolder *> DQMFoldersMap;
    DQMFoldersMap dqmFolders_;
    
  private:
    TTimeStamp            *firstUpdate_;
    TTimeStamp            *lastUpdate_;
    TTimeStamp            *lastServed_;
    unsigned int           nUpdates_;
    const time_t           readyTime_;
    const unsigned int     expectedUpdates_;
    bool                   wasServedSinceUpdate_;
  }; 



  /**
   * Container class for one snapshot instance of a collection of 
   * collated DQM groups
   *
   * $Author: mommsen $
   * $Revision: 1.15 $
   * $Date: 2010/03/04 11:19:40 $
   */

  class DQMInstance
  {
  public:
    DQMInstance(const int runNumber,
                const int lumiSection,
                const int updateNumber,
                const time_t purgeTime,
                const time_t readyTime,
                const unsigned int expectedUpdates);

    ~DQMInstance();

    inline int getRunNumber() const    { return(runNumber_);}
    inline int getLumiSection() const  { return(lumiSection_);}
    inline int getUpdateNumber() const { return(updateNumber_);}
    inline time_t getPurgeTime() const { return(purgeTime_);}
    inline time_t getReadyTime() const { return(readyTime_);}

    inline TTimeStamp * getFirstUpdate() const { return(firstUpdate_);}
    inline TTimeStamp * getLastUpdate() const  { return(lastUpdate_);}

    bool isReady(time_t now) const;
    bool isStale(time_t now) const;

    void addEvent(const std::string topFolderName, std::auto_ptr<DQMEvent::TObjectTable>);
    double writeFile(std::string filePrefix, bool endRunFlag) const;
    DQMGroup * getDQMGroup(std::string groupName) const;

  protected:
    typedef std::map<std::string, DQMGroup *> DQMGroupsMap;
    DQMGroupsMap dqmGroups_;

  private:
    const int              runNumber_;
    const int              lumiSection_;
    const int              updateNumber_;
    TTimeStamp            *firstUpdate_;
    TTimeStamp            *lastUpdate_;
    unsigned int           nUpdates_;
    const time_t           purgeTime_;
    const time_t           readyTime_;
    const unsigned int     expectedUpdates_;
  };

  class DQMGroupDescriptor
  // only used by DQMServiceManager which is used by old SM proxy server
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
