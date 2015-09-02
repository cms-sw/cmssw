#ifndef DQMServices_StreamerIO_DQMMonitoringService_h
#define DQMServices_StreamerIO_DQMMonitoringService_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "boost/filesystem.hpp"

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <sstream>
#include <chrono>
#include <unordered_map>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

/*
 * This service is very similar to the FastMonitoringService in the HLT,
 * except that it is used for monitoring online DQM applications
 */

namespace dqmservices {

using boost::property_tree::ptree;
using edm::StreamID;
using edm::StreamContext;
using edm::GlobalContext;

class DQMMonitoringService {
  public:
    DQMMonitoringService(const edm::ParameterSet &, edm::ActivityRegistry&);
    ~DQMMonitoringService();

    void keepAlive();
    void outputUpdate(ptree& doc);

    void evLumi(GlobalContext const&);
    void evEvent(StreamID const&);
    
    //void makeReport();

  private:
    std::shared_ptr<std::ostream> mstream_;
    ptree doc_;

    long nevents_;
    long last_report_nevents_;
    std::chrono::high_resolution_clock::time_point last_report_time_;
};

} // end-of-namespace

#endif
