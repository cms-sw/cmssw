#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/StreamerInputFile.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"

#include "DQMStreamerReader.h"

#include <fstream>
#include <boost/regex.hpp>
#include <boost/range.hpp>
#include <boost/filesystem.hpp>

namespace edm {

  DQMStreamerReader::DQMStreamerReader(ParameterSet const& pset, InputSourceDescription const& desc): 
      StreamerInputSource(pset, desc),
      runNumber_(pset.getUntrackedParameter<unsigned int> ("runNumber")),
      dqmInputDir_(pset.getUntrackedParameter<std::string> ("dqmInputDir")),
      currentLumiSection_(1),
      totalEventPerLs_(0),
      processedEventPerLs_(0),
      streamerName_(""),
      streamReader_(),
      eventSkipperByID_(EventSkipperByID::create(pset).release()),
      initialNumberOfEventsToSkip_(pset.getUntrackedParameter<unsigned int>("skipEvents", 0)) {
      reset_();   
  }

  DQMStreamerReader::~DQMStreamerReader() {    
  }

  void DQMStreamerReader::reset_() {
    while (!checkNewData(currentLumiSection_)){
      currentLumiSection_ += 1;
    }
    streamerName_ = getDataFile(currentLumiSection_); 
    std::cout << "************ reset: using fileName " << streamerName_ << std::endl;     
    openNewFile(streamerName_);
  }

  void DQMStreamerReader::openNewFile(std::string newStreamerFile_){
    std::cout << "----- openNewFile:  " << newStreamerFile_ << std::endl;     
    processedEventPerLs_ = 0;
    streamReader_ = std::unique_ptr<StreamerInputFile>(new StreamerInputFile(newStreamerFile_, eventSkipperByID_));

    InitMsgView const* header = getHeader();
    deserializeAndMergeWithRegistry(*header, false);
    if(initialNumberOfEventsToSkip_) {
      skip(initialNumberOfEventsToSkip_);
    }
  }

  DQMStreamerReader::DQMJSON DQMStreamerReader::loadJSON(int lumi)
  {
    DQMJSON dqmjson;
    dqmjson.load(make_path(lumi));
    return dqmjson;
  }

  std::string DQMStreamerReader::getDataFile(int lumi){
    DQMJSON dqmjson = loadJSON(lumi);
    std::string datafile = dqmjson.datafilename;
    std::string newfilename;
    newfilename = dqmInputDir_ 
      + "/run" + to_padded_string(runNumber_, run_min_length)
      + "/" + datafile;
    return newfilename;
  }

  bool DQMStreamerReader::isEndOfRun()
  {
    std::string endOfRunFileName_;
    endOfRunFileName_ = dqmInputDir_
                  + "/run" + to_padded_string(runNumber_, run_min_length)
                  + "/run" + to_padded_string(runNumber_, run_min_length)
                  + "_EOR.jsn";
    return boost::filesystem::exists(endOfRunFileName_);
  }

  bool DQMStreamerReader::checkNewData(int lumi){ 
    if ( !boost::filesystem::exists(make_path(lumi))){
      std::cout << "Json file " << make_path(lumi) << " not found!" << std::endl;
      return false;
    }
    DQMJSON dqmjson = loadJSON(lumi);
    totalEventPerLs_ = dqmjson.n_events;
    if (totalEventPerLs_ > 0){
      std::cout << " CheckNewData: JSON file " << make_path(lumi) << " has " << totalEventPerLs_ << " events" << std::endl; 
      return true;
    }else{
      std::cout << " WARNING: CheckNewData: JSON file " << make_path(lumi) << " has no events" << std::endl; 
      return false;
    }
    return false;
  }
    
  std::string DQMStreamerReader::to_padded_string(int n, std::size_t min_length)
  {
    std::string ret(std::to_string(n));
    
    if (ret.length() >= min_length)
      return ret;
    
    return std::string(min_length - ret.length(), '0') + ret;
  }

  std::string DQMStreamerReader::make_path(lumisection_t lumisection)
  {
    std::string jsonFileName_;
    jsonFileName_ = dqmInputDir_ 
      + "/run" + to_padded_string(runNumber_, run_min_length)
      + "/run" + to_padded_string(runNumber_, run_min_length)
      + "_ls" + to_padded_string(lumisection, lumisection_min_length)
      + ".jsn"; 
    return jsonFileName_;
 }

  bool DQMStreamerReader::checkNextEvent() {
    if ( processedEventPerLs_ > 0){
      std::cout << "***** checkNextEvent: at least one processed event: check if new LS exist" << std::endl;     
      if ( checkNextLS() ){
	std::cout << "***** checkNextEvent: at least one event has been processed from current LS and a new LS has been found " << std::endl;
      }
    }else{
      std::cout << "****** checkNextEvent: no event processed in LS "  << currentLumiSection_ << std::endl;
    }

    EventMsgView const* eview = getNextEvent();

    if (eview == nullptr) {
      // no more events in this lumi, wait
      std::cout << "NO MORE EVENTS -- wait" << std::endl;  

      while(!checkNextLS()){
	if (isEndOfRun()){
	  std::cout << "Run " << runNumber_ << " is finished" << std::endl;
	  return 0;
	}else{
	  std::cout << "No event available ... wait for next LS"<< std::endl;
	  usleep(100000);
	}
      }
    }
 
    if (newHeader()) {
      // A new file has been opened and we must compare Headers here !!
      //Get header/init from reader
      InitMsgView const* header = getHeader();
      deserializeAndMergeWithRegistry(*header, true);
    }
   deserializeEvent(*eview);
    return true;
  }

  bool DQMStreamerReader::checkNextLS() {
    std::cout << "************ checkNextLS: check for LS " << currentLumiSection_ + 1 << std::endl;     
    int nextLS = currentLumiSection_ + 1; 
    if (checkNewData(nextLS) ){
      std::cout << "New LS found: LS # " << nextLS  << std::endl;
      closeFile_();
      currentLumiSection_ +=1;
      streamerName_ = getDataFile(currentLumiSection_); 
      openNewFile(streamerName_);
      return true;
    }else{
      std::cout << "No new LS found: processing current LS " << currentLumiSection_ << std::endl;
      return false;
    }
  }

  void DQMStreamerReader::skip(int toSkip) {
    for(int i = 0; i != toSkip; ++i) {
      EventMsgView const* evMsg = getNextEvent();
      if(evMsg == nullptr)  {
        return;
      }
      // If the event would have been skipped anyway, don't count it as a skipped event.
      if(eventSkipperByID_ && eventSkipperByID_->skipIt(evMsg->run(), evMsg->lumi(), evMsg->event())) {
        --i;
      }
    }
  }

  void DQMStreamerReader::closeFile_() {
    std::cout << "************ closeFile_ " << streamerName_ << std::endl;     
    if(streamReader_.get() != nullptr) streamReader_->closeStreamerFile();
  }

  bool DQMStreamerReader::newHeader() {
    std::cout << "************ newHeader " << std::endl;     
    return streamReader_->newHeader();
  }

  InitMsgView const* DQMStreamerReader::getHeader() {
    std::cout << "************ getHeader " << std::endl;     
    InitMsgView const* header = streamReader_->startMessage();

    if(header->code() != Header::INIT) { //INIT Msg
      throw Exception(errors::FileReadError, "DQMStreamerReader::readHeader")
        << "received wrong message type: expected INIT, got "
        << header->code() << "\n";
   } 
    return header;
  }

  EventMsgView const* DQMStreamerReader::getNextEvent() {
    if (!streamReader_->next()) {
      return nullptr;
    }
    processedEventPerLs_ += 1;
    std::cout << "************  getNextEvent " << " event processed " << processedEventPerLs_ << std::endl;      
    return streamReader_->currentRecord();
  }

  void
  DQMStreamerReader::fillDescriptions(ConfigurationDescriptions& descriptions) {
    ParameterSetDescription desc;
    desc.setComment("Reads events from streamer files.");
    desc.addUntracked<unsigned int>("runNumber")
      ->setComment("Run number passed via configuration file");
    desc.addUntracked<std::string>("dqmInputDir")
      ->setComment("Directory where DQM files will appear");
    desc.addUntracked<unsigned int>("skipEvents", 0U)
        ->setComment("Skip the first 'skipEvents' events that otherwise would have been processed.");
    desc.addUntracked<bool>("inputFileTransitionsEachEvent", false);
    StreamerInputSource::fillDescription(desc);
    EventSkipperByID::fillDescription(desc);
    descriptions.add("source", desc);
  }
} 

