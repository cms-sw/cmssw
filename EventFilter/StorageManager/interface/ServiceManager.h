#ifndef _SERVICEMANAGER_H_
#define _SERVICEMANAGER_H_

// $Id: ServiceManager.h,v 1.4 2008/05/04 12:34:05 biery Exp $

#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Framework/interface/EventSelector.h"

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include <EventFilter/StorageManager/interface/StreamService.h>
#include <EventFilter/StorageManager/interface/InitMsgCollection.h>

#include <boost/shared_ptr.hpp>
#include <vector>
#include <list>
#include <string>
#include <map>

namespace edm 
{
  
  typedef std::vector<boost::shared_ptr<StreamService> >            Streams;
  typedef std::vector<boost::shared_ptr<StreamService> >::iterator  StreamsIterator;
  
  
  class ServiceManager {
    
  public:  
    
    explicit ServiceManager(const std::string& config);
    ~ServiceManager(); 
    
    void stop(); 
    
    void manageInitMsg(std::string catalog, uint32 disks, std::string sourceId, InitMsgView& init_message, stor::InitMsgCollection& initMsgCollection);
    
    void manageEventMsg(EventMsgView& msg);
    
    std::list<std::string>& get_filelist();
    std::list<std::string>& get_currfiles();
    std::vector<uint32>& get_storedEvents();
    std::vector<std::string>& get_storedNames();

    std::map<std::string, Strings> getStreamSelectionTable();
    
  private:   
    void collectStreamerPSets(const std::string& config);        
    
    std::vector<ParameterSet>              outModPSets_;
    Streams                                managedOutputs_;  
    boost::shared_ptr<edm::EventSelector>  eventSelector_;
    std::list<std::string>                 filelist_;
    std::list<std::string>                 currfiles_;
    Strings                                psetHLTOutputLabels_;
    std::vector<uint32>                    outputModuleIds_;
    std::vector<uint32>                    storedEvents_;
    std::vector<std::string>              storedNames_;
  };
  
}//edm-namespace

#endif
