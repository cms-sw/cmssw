// -*- C++ -*-
//
// Package:     DQMServices/Components
// Class  :     fillJson
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Thu, 08 Nov 2018 21:20:03 GMT
//

// system include files
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/format.hpp>

#include <string>
#include <sstream>
#include <filesystem>

// user include files
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Components/interface/fillJson.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"
#include "FWCore/Utilities/interface/Exception.h"

boost::property_tree::ptree dqmfilesaver::fillJson(int run,
                                                   int lumi,
                                                   const std::string& dataFilePathName,
                                                   const std::string& transferDestinationStr,
                                                   const std::string& mergeTypeStr,
                                                   evf::FastMonitoringService* fms) {
  namespace bpt = boost::property_tree;

  bpt::ptree pt;

  int hostnameReturn;
  char host[32];
  hostnameReturn = gethostname(host, sizeof(host));
  if (hostnameReturn == -1)
    throw cms::Exception("fillJson") << "Internal error, cannot get host name";

  int pid = getpid();
  std::ostringstream oss_pid;
  oss_pid << pid;

  int nProcessed = fms ? (fms->getEventsProcessedForLumi(lumi)) : -1;

  // Stat the data file: if not there, throw
  std::string dataFileName;
  struct stat dataFileStat;
  dataFileStat.st_size = 0;
  if (nProcessed) {
    if (stat(dataFilePathName.c_str(), &dataFileStat) != 0)
      throw cms::Exception("fillJson") << "Internal error, cannot get data file: " << dataFilePathName;
    // Extract only the data file name from the full path
    dataFileName = std::filesystem::path(dataFilePathName).filename().string();
  }
  // The availability test of the FastMonitoringService was done in the ctor.
  bpt::ptree data;
  bpt::ptree processedEvents, acceptedEvents, errorEvents, bitmask, fileList, fileSize, inputFiles, fileAdler32,
      transferDestination, mergeType, hltErrorEvents;

  processedEvents.put("", nProcessed);  // Processed events
  acceptedEvents.put("", nProcessed);   // Accepted events, same as processed for our purposes

  errorEvents.put("", 0);                               // Error events
  bitmask.put("", 0);                                   // Bitmask of abs of CMSSW return code
  fileList.put("", dataFileName);                       // Data file the information refers to
  fileSize.put("", dataFileStat.st_size);               // Size in bytes of the data file
  inputFiles.put("", "");                               // We do not care about input files!
  fileAdler32.put("", -1);                              // placeholder to match output json definition
  transferDestination.put("", transferDestinationStr);  // SM Transfer destination field
  mergeType.put("", mergeTypeStr);                      // Merging type for merger and transfer services
  hltErrorEvents.put("", 0);                            // Error events

  data.push_back(std::make_pair("", processedEvents));
  data.push_back(std::make_pair("", acceptedEvents));
  data.push_back(std::make_pair("", errorEvents));
  data.push_back(std::make_pair("", bitmask));
  data.push_back(std::make_pair("", fileList));
  data.push_back(std::make_pair("", fileSize));
  data.push_back(std::make_pair("", inputFiles));
  data.push_back(std::make_pair("", fileAdler32));
  data.push_back(std::make_pair("", transferDestination));
  data.push_back(std::make_pair("", mergeType));
  data.push_back(std::make_pair("", hltErrorEvents));

  pt.add_child("data", data);

  if (fms == nullptr) {
    pt.put("definition", "/fakeDefinition.jsn");
  } else {
    // The availability test of the EvFDaqDirector Service was done in the ctor.
    std::filesystem::path outJsonDefName{
        edm::Service<evf::EvFDaqDirector>()->baseRunDir()};  //we assume this file is written bu the EvF Output module
    outJsonDefName /= (std::string("output_") + oss_pid.str() + std::string(".jsd"));
    pt.put("definition", outJsonDefName.string());
  }

  char sourceInfo[64];  //host and pid information
  sprintf(sourceInfo, "%s_%d", host, pid);
  pt.put("source", sourceInfo);

  return pt;
}
