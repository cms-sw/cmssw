#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "IOPool/Streamer/interface/StreamerOutputService.h"

#include "boost/shared_ptr.hpp"

#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <iterator>

using namespace std;
//using namespace edm;

namespace edm {

 class StreamerOutSrvcManager {

 public:  

  explicit StreamerOutSrvcManager(const std::string& config);
  ~StreamerOutSrvcManager(); 

  void stop(); 

  /** Handles arrival of New Init Message */
  void manageInitMsg(unsigned long maxFileSize, double highWaterMark,
		  std::string path, std::string mpath, InitMsgView& init_message);

  /** mages event messages */
  void manageEventMsg(EventMsgView& msg);

 private:   
  void collectStreamerPSets(const std::string& config);        
  //Store References ?? 
  std::vector<edm::ParameterSet> outModPSets_;
  std::vector<edm::StreamerOutputService*> managedOutputs_;  
 };

}//edm-namespace

