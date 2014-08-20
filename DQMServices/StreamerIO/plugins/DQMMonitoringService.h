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

class DQMMonitoringService {
  public:
    DQMMonitoringService(const edm::ParameterSet &, edm::ActivityRegistry&);
    ~DQMMonitoringService();

    void registerExtra(std::string name, ptree data);
    void reportLumiSection(int run, int lumi);
    void reportEvents(int nevts);

  private:
    boost::filesystem::path json_path_;
    std::string hostname_;
    std::string tag_;
    int fseq_;
    long nevents_;

    ptree extra_;
    ptree ps_info_;

    long last_report_nevents_;
    std::chrono::high_resolution_clock::time_point last_report_time_;

    void reportLumiSectionUnsafe(int run, int lumi);

    void fillProcessInfoCmdline();
    void fillProcessInfoStatus();
    std::string hackoutTheStdErr();
};

} // end-of-namespace

#endif
