#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/file.h>
#include <unistd.h>
#include <vector>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem/fstream.hpp>

#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/FEDInterface/interface/fed_header.h"
#include "EventFilter/FEDInterface/interface/fed_trailer.h"

#include "EventFilter/Utilities/interface/FileIO.h"

#include "FedRawDataInputSource.h"
#include "FastMonitoringService.h"
#include "EvFDaqDirector.h"

FedRawDataInputSource::FedRawDataInputSource(edm::ParameterSet const& pset,
                                             edm::InputSourceDescription const& desc) :
  edm::RawInputSource(pset, desc),
  getLSFromFilename_(
                     pset.getUntrackedParameter<bool> ("getLSFromFilename", true)),
  testModeNoBuilderUnit_(edm::Service<evf::EvFDaqDirector>()->getTestModeNoBuilderUnit()),
  eventChunkSize_(pset.getUntrackedParameter<unsigned int> ("eventChunkSize",16)),
  runNumber_(pset.getUntrackedParameter<unsigned int> ("runNumber")),
  daqProvenanceHelper_(edm::TypeID(typeid(FEDRawDataCollection))),
  formatVersion_(0),
  fileIndex_(0),
  fileStream_(0),
  workDirCreated_(false),
  eventID_(),
  lastOpenedLumi_(0),
  currentDataDir_(""),
  currentInputJson_(""),
  currentInputEventCount_(0),
  eorFileSeen_(false),
  buffer_left(0),
  data_buffer(new unsigned char[1024 * 1024 * eventChunkSize_])

{
  gethostname(thishost, 255); 
  edm::LogInfo("FedRawDataInputSource") << "test mode: "
                                        << testModeNoBuilderUnit_ << ", read-ahead chunk size: " << eventChunkSize_
                                        << " on host " << thishost;

  daqProvenanceHelper_.daqInit(productRegistryUpdate(), processHistoryRegistryForUpdate());
  findAllDirectories();
  setNewRun();
  setRunAuxiliary(
                  new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(),
					edm::Timestamp::invalidTimestamp()));
}

FedRawDataInputSource::~FedRawDataInputSource() {
	if (fileStream_)
		fclose(fileStream_);
	fileStream_ = 0;
}

void FedRawDataInputSource::findAllDirectories() {
  
  std::string rootFUDirectory = edm::Service<evf::EvFDaqDirector>()->baseDir();
  // is this really necessary ? Should not be as the process start is now *triggered* by the appearence of this directory...
  boost::filesystem::directory_iterator itEnd;
  bool foundInFu = false;
  for (boost::filesystem::directory_iterator it(rootFUDirectory); it
         != itEnd; ++it) {
    std::string::size_type pos = std::string::npos;
    if (boost::filesystem::is_directory(it->path())
        && std::string::npos != (pos = it->path().string().find("run"))){
      std::string rnString = it->path().string().substr(pos+3);
      if(runNumber_ == (unsigned int)atoi(rnString.c_str())) {foundInFu=true; localRunBaseDirectory_=it->path(); break;}
    }
  }
  if(!foundInFu) throw cms::Exception("LogicError") << "Run directory for run " << runNumber_ << " not found on Fu disk " 
                                                          << rootFUDirectory;

  // store the path to the BU (input) dir
  buRunDirectory_ = boost::filesystem::path(edm::Service<evf::EvFDaqDirector>()->buBaseDir());
  //  buRunDirectory_ /= localRunBaseDirectory_.filename();
  
  
  if (!boost::filesystem::exists(buRunDirectory_)) 
    throw cms::Exception("LogicError") << "Run directory for run " << runNumber_ << " not found on Bu disk " 
                                       << buRunDirectory_.string();
  edm::LogInfo("FedRawDataInputSource") << "Getting data from "
                                        << buRunDirectory_.string();

}

bool FedRawDataInputSource::checkNextEvent() {

  FRDEventHeader_V2 eventHeader;

  if (!getEventHeaderFromBuffer(&eventHeader)) {
    // run has ended
    resetLuminosityBlockAuxiliary();
    return false;
  }

  assert(eventHeader.version_ > 1);
  formatVersion_ = eventHeader.version_;

  //same lumi, or new lumi detected in file (old mode)
  if (!getLSFromFilename_) {
    //get new lumi from file header
    if (!luminosityBlockAuxiliary()
        || luminosityBlockAuxiliary()->luminosityBlock()
        != eventHeader.lumi_) {
      lastOpenedLumi_ = eventHeader.lumi_;
      resetLuminosityBlockAuxiliary();
      timeval tv;
      gettimeofday(&tv, 0);
      edm::Timestamp lsopentime(
                                (unsigned long long) tv.tv_sec * 1000000
                                + (unsigned long long) tv.tv_usec);
      edm::LuminosityBlockAuxiliary* luminosityBlockAuxiliary =
        new edm::LuminosityBlockAuxiliary(runAuxiliary()->run(),
                                          eventHeader.lumi_, lsopentime,
                                          edm::Timestamp::invalidTimestamp());
      setLuminosityBlockAuxiliary(luminosityBlockAuxiliary);
    }

  } else {

    //new lumi from directory name
    if (!luminosityBlockAuxiliary()
        || luminosityBlockAuxiliary()->luminosityBlock()
        != lastOpenedLumi_) {
      resetLuminosityBlockAuxiliary();
      
      timeval tv;
      gettimeofday(&tv, 0);
      edm::Timestamp lsopentime(
                                (unsigned long long) tv.tv_sec * 1000000
                                + (unsigned long long) tv.tv_usec);
      edm::LuminosityBlockAuxiliary* luminosityBlockAuxiliary =
        new edm::LuminosityBlockAuxiliary(runAuxiliary()->run(),
                                          lastOpenedLumi_, lsopentime,
                                          edm::Timestamp::invalidTimestamp());
      setLuminosityBlockAuxiliary(luminosityBlockAuxiliary);
    }
  }

  eventID_ = edm::EventID(eventHeader.run_, lastOpenedLumi_,
                          eventHeader.event_);

  setEventCached();

  return true;

}

void FedRawDataInputSource::read(edm::EventPrincipal& eventPrincipal) 
{
  currentInputEventCount_--;
  std::auto_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  edm::Timestamp tstamp = fillFEDRawDataCollection(rawData);
  
  edm::EventAuxiliary aux(eventID_, processGUID(), tstamp, true,
                          edm::EventAuxiliary::PhysicsTrigger);
    makeEvent(eventPrincipal, aux);
  
  edm::WrapperOwningHolder edp(
                               new edm::Wrapper<FEDRawDataCollection>(rawData),
                               edm::Wrapper<FEDRawDataCollection>::getInterface());
  
  eventPrincipal.put(daqProvenanceHelper_.constBranchDescription_, edp,
                     daqProvenanceHelper_.dummyProvenance_);
  
  return;
}

bool FedRawDataInputSource::eofReached() const {
  if (fileStream_ == 0)
    return true;
  
  int c;
  c = fgetc(fileStream_);
  ungetc(c, fileStream_);
  
  return (c == EOF);
}

edm::Timestamp FedRawDataInputSource::fillFEDRawDataCollection(std::auto_ptr<FEDRawDataCollection>& rawData) {
  edm::Timestamp tstamp;
  uint32_t eventSize = 0;
  uint32_t paddingSize = 0;
  if (formatVersion_ >= 3) {
    eventSize = getEventSizeFromBuffer();
    paddingSize = getPaddingSizeFromBuffer();
  }
  uint32_t fedSizes[1024];
  eventSize += fillFedSizesFromBuffer(fedSizes);
  
  /*
    if (formatVersion_ < 3) {
    for (unsigned int i = 0; i < 1024; i++)
    eventSize += fedSizes[i];
    }
  */

  unsigned int gtpevmsize = fedSizes[FEDNumbering::MINTriggerGTPFEDID];
  if (gtpevmsize > 0)
    evf::evtn::evm_board_setformat(gtpevmsize);
  
  //todo put this in a separate function
  if (buffer_left < eventSize)
    checkIfBuffered();
  char* event = (char *) (data_buffer + buffer_cursor);
  buffer_left -= eventSize;
  buffer_cursor += eventSize;
  
  while (eventSize > 0) {
    eventSize -= sizeof(fedt_t);
    const fedt_t* fedTrailer = (fedt_t*) (event + eventSize);
    const uint32_t fedSize = FED_EVSZ_EXTRACT(fedTrailer->eventsize) << 3; //trailer length counts in 8 bytes
    eventSize -= (fedSize - sizeof(fedh_t));
    const fedh_t* fedHeader = (fedh_t *) (event + eventSize);
    const uint16_t fedId = FED_SOID_EXTRACT(fedHeader->sourceid);
    if (fedId == FEDNumbering::MINTriggerGTPFEDID) {
      const uint64_t gpsl = evf::evtn::getgpslow(
                                                 (unsigned char*) fedHeader);
      const uint64_t gpsh = evf::evtn::getgpshigh(
                                                  (unsigned char*) fedHeader);
      tstamp = edm::Timestamp(
                              static_cast<edm::TimeValue_t> ((gpsh << 32) + gpsl));
    }
    FEDRawData& fedData = rawData->FEDData(fedId);
    fedData.resize(fedSize);
    memcpy(fedData.data(), event + eventSize, fedSize);
  }
  assert(eventSize == 0);
  buffer_left -= paddingSize;
  buffer_cursor += paddingSize;
	//	delete event;
  return tstamp;
}

bool FedRawDataInputSource::openNextFile() {
  // @@EM !!!!!!!!!!!!!!    this function is dead wood
  createWorkingDirectory();
  boost::filesystem::path nextFile;
  //  boost::filesystem::path nextFile = buRunDirectory_;

  //  std::ostringstream fileName;
  //   fileName << std::setfill('0') << std::setw(16) << fileIndex_++ << "_"
  //            << thishost << "_" << getpid() << ".raw";
  //   nextFile /= fileName.str();
  
  //   openFile(nextFile);//closes previous file

  while (!searchForNextFile(nextFile) && !eorFileSeen_) {
    std::cout << "No file for me... sleep and try again..." << std::endl;
    usleep(100000);
  }

  return (fileStream_ != 0 || !eorFileSeen_);
}

void FedRawDataInputSource::openFile(boost::filesystem::path const& nextFile) {

  if (fileStream_) {
    fclose(fileStream_);
    fileStream_ = 0;
    
    if (!testModeNoBuilderUnit_) {
      boost::filesystem::remove(openFile_); // won't work in case of forked children
    } else {
      renameToNextFree();
    }
    
  }

  const int fileDescriptor = open(nextFile.c_str(), O_RDONLY);
  if (fileDescriptor != -1) {
    fileStream_ = fdopen(fileDescriptor, "rb");
    openFile_ = nextFile;
  }
  edm::LogInfo("FedRawDataInputSource") << " tried to open file.. " << nextFile << " fd:"
            << fileDescriptor;
}

void FedRawDataInputSource::renameToNextFree()
{
  boost::filesystem::path fileToRename(openFile_);

  unsigned int jumpLS =
    edm::Service<evf::EvFDaqDirector>()->getJumpLS();
  unsigned int jumpIndex =
    edm::Service<evf::EvFDaqDirector>()->getJumpIndex();

  string path = formatRawFilePath(jumpLS,jumpIndex);
  edm::LogInfo("FedRawDataInputSource") << "Instead of delete, RENAME: " << openFile_ 
                                        << " to: " << path;
  int rc = rename(openFile_.string().c_str(), path.c_str());
  if (rc != 0) {
    edm::LogError("FedRawDataInputSource") << "RENAME RAW FAILED!";
  }
  
  edm::LogInfo("FedRawDataInputSource") << "Also rename json: " << openFile_ << " to: " << path;
  string sourceJson = formatJsnFilePath(jumpLS - 2,jumpIndex);
  string destJson = formatJsnFilePath(jumpLS,jumpIndex);
  rc = rename(sourceJson.c_str(), destJson.c_str());

  if (rc != 0) {
    edm::LogError("FedRawDataInputSource") << "RENAME JSON FAILED!";
  }
}

bool FedRawDataInputSource::searchForNextFile(boost::filesystem::path const& nextFile) {

  std::stringstream ss;
  unsigned int ls = lastOpenedLumi_;
  unsigned int index;
  unsigned int initialLS = ls;
  
  if(currentInputEventCount_!=0){
    throw cms::Exception("RuntimeError") << "Went to search for next file but according to BU more events in " 
                                         << currentInputJson_.string();
  }
  boost::filesystem::remove(currentInputJson_); //the content of this file is now accounted for in the output json
  
  edm::LogInfo("FedRawDataInputSource") << "Asking for next file... to the DaqDirector";
  evf::FastMonitoringService*fms = (evf::FastMonitoringService *) (edm::Service<evf::MicroStateService>().operator->());
  fms->startedLookingForFile();
  bool fileIsOKToGrab = edm::Service<evf::EvFDaqDirector>()->updateFuLock(ls,index, eorFileSeen_);

  if (fileIsOKToGrab) {

    string path = formatRawFilePath(ls,index);
    edm::LogInfo("FedRawDataInputSource") << "The director says to grab: " << path;

    fms->stoppedLookingForFile();
    edm::LogInfo("FedRawDataInputSource") << "grabbing next file, setting last seen lumi to LS = " << ls;
    boost::filesystem::path theFileToGrab(path);
    assert(grabNextFile(theFileToGrab, nextFile));
    if (getLSFromFilename_)
      lastOpenedLumi_ = ls;
    return true;
      
  } else if (ls > initialLS && !eorFileSeen_) {
    // ls was increased, so some EoL jsn files were seen, without new data files
    edm::LogInfo("FedRawDataInputSource") << "EoL jsn file(s) seen! Current LS is: " << ls;
    resetLuminosityBlockAuxiliary();
    timeval tv;
    gettimeofday(&tv, 0);
    edm::Timestamp lsopentime(
                              (unsigned long long) tv.tv_sec * 1000000
                              + (unsigned long long) tv.tv_usec);
    edm::LuminosityBlockAuxiliary* luminosityBlockAuxiliary =
      new edm::LuminosityBlockAuxiliary(runAuxiliary()->run(), ls,
                                        lsopentime, edm::Timestamp::invalidTimestamp());
    setLuminosityBlockAuxiliary(luminosityBlockAuxiliary);
          
    return false;
    
  } else {
    edm::LogInfo("FedRawDataInputSource") << "The DAQ Director has nothing for me! ";
    if (eorFileSeen_) {
      edm::LogInfo("FedRawDataInputSource") << "...and it has seen the end of run file! - running some cleanup";
      if (fileStream_) {
        edm::LogInfo("FedRawDataInputSource") << "Closing input stream ";
        fclose(fileStream_);
        fileStream_ = 0;
        if (!testModeNoBuilderUnit_) {
          edm::LogInfo("FedRawDataInputSource") << "Removing last input file used : " << openFile_.string();
          boost::filesystem::remove(openFile_); // won't work in case of forked children
        }
      }
    }
    return false;
  }
}

bool FedRawDataInputSource::grabNextFile(boost::filesystem::path& file,
                                         boost::filesystem::path const& nextFile) {
  try {
    // assemble json path on /hlt/data
    boost::filesystem::path nextFileJson = workingDirectory_;
    boost::filesystem::path jsonSourcePath(file);
    boost::filesystem::path jsonDestPath(nextFileJson);
    boost::filesystem::path jsonExt(".jsn");
    jsonSourcePath.replace_extension(jsonExt);
    boost::filesystem::path jsonTempPath(jsonDestPath);

    std::ostringstream fileNameWithPID;
    fileNameWithPID << jsonSourcePath.stem().string() << "_pid"
                    << std::setfill('0') << std::setw(5) << getpid() << ".jsn";
    boost::filesystem::path filePathWithPID(fileNameWithPID.str());
    jsonTempPath /= filePathWithPID;
    edm::LogInfo("FedRawDataInputSource") << " JSON rename " << jsonSourcePath << " to "
                                          << jsonTempPath;
    //COPY JSON
    string mvCmd = (testModeNoBuilderUnit_? "cp " : "mv ") 
      + jsonSourcePath.string() 
      + " "
      + jsonTempPath.string();
    edm::LogInfo("FedRawDataInputSource")<< " Running cmd = " << mvCmd;
    int rc = system(mvCmd.c_str()); // horrible!!! let's find something better !!!
    currentInputJson_ = jsonTempPath; // store location for later deletion.
    boost::filesystem::ifstream ij(jsonTempPath);
    Json::Value deserializeRoot;
    DataPoint dp;
    if(!reader_.parse(ij,deserializeRoot)){
      throw std::runtime_error("Cannot deserialize input JSON file");
    }
    else{
      dp.deserialize(deserializeRoot);
      std::string data = dp.getData()[0];
      currentInputEventCount_=atoi(data.c_str()); //all this is horrible...
    }
    
    //std::cout << " return code = " << rc << std::endl;
    if (rc != 0) {
      throw std::runtime_error("Cannot copy JSON file, rc is != 0!");
    }
    openFile(file);
    
    return true;
  }

  catch (const boost::filesystem::filesystem_error& ex) {
    // Input dir gone?
    edm::LogError("FedRawDataInputSource") << " - grabNextFile BOOST FILESYSTEM ERROR CAUGHT: " << ex.what()
                  << " - Maybe the BU run dir disappeared? Ending process with code 0...";
    _exit(0);
  } catch (std::runtime_error e) {
    // Another process grabbed the file and NFS did not register this
     edm::LogError("FedRawDataInputSource") << " - grabNextFile runtime Exception: " << e.what() << std::endl;
  } catch (std::exception e) {
    // BU run directory disappeared?
    edm::LogError("FedRawDataInputSource") << " - grabNextFileSOME OTHER EXCEPTION OCCURED!!!! ->" << e.what()
                                           << std::endl;
  }
  return false;
}

/* this functionality now assumed by the DaqDirector */

// bool FedRawDataInputSource::runEnded() const {
// 	boost::filesystem::path endOfRunMarker = buRunDirectory_;
// 	endOfRunMarker /= "EndOfRun.jsn";
// 	return boost::filesystem::exists(endOfRunMarker);
// }

void FedRawDataInputSource::preForkReleaseResources() {
}

void FedRawDataInputSource::postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource>) {
  createWorkingDirectory();
  InputSource::rewind();
  setRunAuxiliary(
                  new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(),
					edm::Timestamp::invalidTimestamp()));
}

void FedRawDataInputSource::rewind_() {
}

void FedRawDataInputSource::createWorkingDirectory() {

  if(workDirCreated_) return;
  //this function is moot since all processes share the same working directory which is the local run base 
  workingDirectory_ = localRunBaseDirectory_;
  // workingDirectory_ /= "open";
  
  // 	if (!openDirFound) {
  // 		boost::filesystem::create_directories(workingDirectory_);
  // 	}

  workDirCreated_ = true;

  // also create MON directory

  boost::filesystem::path monDirectory = localRunBaseDirectory_;
  monDirectory /= "mon";
  
  bool foundMonDir = false;
  if (boost::filesystem::is_directory(monDirectory))
    foundMonDir = true;
  if (!foundMonDir) {
    edm::LogInfo("FedRawDataInputSource") << "mon directory not found - creating";
    boost::filesystem::create_directories(monDirectory);
  }

}

uint32_t FedRawDataInputSource::getEventSizeFromBuffer() {
  if (buffer_left < sizeof(uint32_t))
    checkIfBuffered();
  uint32_t retval = *(uint32_t*) (data_buffer + buffer_cursor);
  buffer_left -= sizeof(uint32_t);
  buffer_cursor += sizeof(uint32_t);
  return retval;
}

uint32_t FedRawDataInputSource::getPaddingSizeFromBuffer() {
  if (buffer_left < sizeof(uint32_t))
    checkIfBuffered();
  uint32_t retval = *(uint32_t*) (data_buffer + buffer_cursor);
  buffer_left -= sizeof(uint32_t);
  buffer_cursor += sizeof(uint32_t);
  return retval;
}

uint32_t FedRawDataInputSource::fillFedSizesFromBuffer(uint32_t *fedSizes) {
  if (buffer_left < sizeof(uint32_t) * 1024)
    checkIfBuffered();
  memcpy((void*) fedSizes, (void*) (data_buffer + buffer_cursor),
         sizeof(uint32_t) * 1024);
  uint32_t eventSize = 0;
  if (formatVersion_ < 3) {
    for (unsigned int i = 0; i < 1024; i++)
      eventSize += fedSizes[i];
  }
  buffer_left -= sizeof(uint32_t) * 1024;
  buffer_cursor += sizeof(uint32_t) * 1024;
  return eventSize;
}

bool FedRawDataInputSource::getEventHeaderFromBuffer(FRDEventHeader_V2 *eventHeader) {
  if (buffer_left < sizeof(uint32_t) * 4)
    if (!checkIfBuffered())
      return false;
  memcpy((void*) eventHeader, (void*) (data_buffer + buffer_cursor),
         sizeof(uint32_t) * 4);
  assert(eventHeader->version_ > 1);
  formatVersion_ = eventHeader->version_;
  buffer_left -= sizeof(uint32_t) * 4;
  buffer_cursor += sizeof(uint32_t) * 4;
  return true;
}

bool FedRawDataInputSource::checkIfBuffered() {
  //this function is called when we reach the end of the buffer (i.e. bytes to read are more than bytes left in buffer)
  if (eofReached() && !openNextFile())
    return false; //should only happen when requesting the next event header and the run is over
  if (buffer_left == 0) { //in the rare case the last byte barely fit
    uint32_t chunksize = 1024 * 1024 * eventChunkSize_;
    buffer_left = fread((void*) data_buffer, sizeof(unsigned char),
                        chunksize, fileStream_); //reads a maximum of chunksize bytes from file into buffer
  } else { //refill the buffer after having moved the buffer_left bytes at the head
    uint32_t chunksize = 1024 * 1024 * eventChunkSize_ - buffer_left;
    memcpy((void*) data_buffer, data_buffer + buffer_cursor, buffer_left); //this copy could be avoided
    buffer_left += fread((void*) (data_buffer + buffer_left),
                         sizeof(unsigned char), chunksize, fileStream_);
  }
  buffer_cursor = 0; // reset the cursor at the beginning of the buffer
  return (buffer_left != 0);
}

// @@EM !!! can be done a lot better than this 
std::string FedRawDataInputSource::formatRawFilePath(unsigned int ls, unsigned int index){
  stringstream ss;
  ss << buRunDirectory_.string() << "/run" << std::setfill('0') << std::setw(6) << runNumber_
     << "_ls" << std::setfill('0') << std::setw(4) << ls 
     << "_index" << std::setfill('0') << std::setw(6) << index << ".raw";
  return ss.str();
}
std::string FedRawDataInputSource::formatJsnFilePath(unsigned int ls, unsigned int index){
  stringstream ss;
  ss << buRunDirectory_.string() << "/run" << std::setfill('0') << std::setw(6) << runNumber_
     << "_ls" << std::setfill('0') << std::setw(4) << ls 
     << "_index" << std::setfill('0') << std::setw(6) << index << ".jsn";
  return ss.str();
}

// define this class as an input source
DEFINE_FWK_INPUT_SOURCE( FedRawDataInputSource);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
