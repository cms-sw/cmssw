#ifndef _SERVICEMANAGER_H_
#define _SERVICEMANAGER_H_

// $Id$

#include "FWCore/ParameterSet/interface/ProcessDesc.h"

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/FRDEventMessage.h"

#include <EventFilter/StorageManager/interface/InitMsgCollection.h>
#include <EventFilter/StorageManager/interface/SMPerformanceMeter.h>
#include <EventFilter/StorageManager/interface/Configuration.h>

#include <boost/shared_ptr.hpp>
#include <vector>
#include <list>
#include <string>
#include <map>

namespace edm 
{
   
  
  class ServiceManager {
    
  public:  
    
    explicit ServiceManager(stor::DiskWritingParams dwParams);
    ~ServiceManager(); 
    
    void start(); 
    void stop(); 
    
    boost::shared_ptr<stor::SMOnlyStats> get_stats();

    std::map<std::string, Strings> getStreamSelectionTable();
    
  private:   
    void collectStreamerPSets(const std::string& config);        
    
    std::vector<ParameterSet>              outModPSets_;
    std::list<std::string>                 filelist_;
    std::list<std::string>                 currfiles_;
    int                                    currentlumi_;
    double                                 timeouttime_;
    double                                 lasttimechecked_;
    int                                    errorStreamPSetIndex_;
    bool                                   errorStreamCreated_;
    unsigned long samples_;
    unsigned long period4samples_;
    stor::SMPerformanceMeter *pmeter_;

    stor::DiskWritingParams diskWritingParams_;
  };
  
}//edm-namespace

#endif
/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
