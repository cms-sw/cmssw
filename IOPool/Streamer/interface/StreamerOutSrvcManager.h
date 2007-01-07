#ifndef _STREAMEROUTSRVCMANAGER_H_
#define _STREAMEROUTSRVCMANAGER_H_

// $Id: StreamerOutSrvcManager.h,v 1.10 2006/12/22 09:48:19 klute Exp $

#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Framework/interface/EventSelector.h"

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/StreamerOutputService.h"
#include "IOPool/Streamer/interface/StreamService.h"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <list>
#include <string>

namespace edm 
{
  
  typedef std::vector<boost::shared_ptr<StreamService> >            Streams;
  typedef std::vector<boost::shared_ptr<StreamService> >::iterator  StreamsIterator;
  
  
  class StreamerOutSrvcManager {
    
  public:  
    
    explicit StreamerOutSrvcManager(const std::string& config);
    ~StreamerOutSrvcManager(); 
    
    void stop(); 
    
    void manageInitMsg(std::string catalog, uint32 disks, std::string souceId, InitMsgView& init_message);
    
    void manageEventMsg(EventMsgView& msg);
    
    std::list<std::string>& get_filelist();
    std::list<std::string>& get_currfiles();
    
  private:   
    void collectStreamerPSets(const std::string& config);        
    
    std::vector<ParameterSet>              outModPSets_;
    Streams                                managedOutputs_;  
    boost::shared_ptr<edm::EventSelector>  eventSelector_;
    std::list<std::string>                 filelist_;
    std::list<std::string>                 currfiles_;
  };
  
}//edm-namespace

#endif
