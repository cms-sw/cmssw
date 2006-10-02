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
  //Currently the fileName parameter is ignored
  void manageInitMsg(std::string fileName, unsigned long maxFileSize, double highWaterMark,
		     std::string path, std::string mpath, std::string catalog, uint32 disks, 
		     InitMsgView& init_message);

  /** mages event messages */
  void manageEventMsg(EventMsgView& msg);

  std::list<std::string>& get_filelist();
  std::list<std::string>& get_currfiles();

 private:   
  void collectStreamerPSets(const std::string& config);        
  //Store References ?? 
  std::vector<edm::ParameterSet> outModPSets_;
  std::vector<edm::StreamerOutputService*> managedOutputs_;  

  // Maintain these lists here, so that we 
  // could return by refs, instead of by value
  std::list<std::string> filelist_;
  std::list<std::string> currfiles_;

 };

}//edm-namespace

