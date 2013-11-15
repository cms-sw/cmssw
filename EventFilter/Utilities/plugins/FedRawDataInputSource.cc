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
  eventChunkSize_(pset.getUntrackedParameter<unsigned int> ("eventChunkSize",16)),
  getLSFromFilename_(pset.getUntrackedParameter<bool> ("getLSFromFilename", true)),
  testModeNoBuilderUnit_(edm::Service<evf::EvFDaqDirector>()->getTestModeNoBuilderUnit()),
  runNumber_(edm::Service<evf::EvFDaqDirector>()->getRunNumber()),
  buInputDir_(edm::Service<evf::EvFDaqDirector>()->buBaseDir()),
  fuOutputDir_(edm::Service<evf::EvFDaqDirector>()->fuBaseDir()),
  daqProvenanceHelper_(edm::TypeID(typeid(FEDRawDataCollection))),
  fileStream_(0),
  eventID_(),
  lastOpenedLumi_(0),
  currentInputJson_(""),
  currentInputEventCount_(0),
  eorFileSeen_(false),
  dataBuffer_(new unsigned char[1024 * 1024 * eventChunkSize_]),
  bufferCursor_(dataBuffer_),
  bufferLeft_(0)
{
  char thishost[256];
  gethostname(thishost, 255); 
  edm::LogInfo("FedRawDataInputSource") << "test mode: "
                                        << testModeNoBuilderUnit_ << ", read-ahead chunk size: " << eventChunkSize_
                                        << " on host " << thishost;

  daqProvenanceHelper_.daqInit(productRegistryUpdate(), processHistoryRegistryForUpdate());
  setNewRun();
  setRunAuxiliary(new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(),
					edm::Timestamp::invalidTimestamp()));
}

FedRawDataInputSource::~FedRawDataInputSource()
{
  if (fileStream_)
    fclose(fileStream_);
  fileStream_ = 0;
}

bool FedRawDataInputSource::checkNextEvent()
{
  if ( ! cacheNextEvent() ) {
    // run has ended
    resetLuminosityBlockAuxiliary();

    if (fileStream_) {
      edm::LogInfo("FedRawDataInputSource") << "Closing input stream ";
      fclose(fileStream_);
      fileStream_ = 0;
      if (!testModeNoBuilderUnit_) {
        edm::LogInfo("FedRawDataInputSource") << "Removing last input file used : " << openFile_.string();
        boost::filesystem::remove(openFile_); // won't work in case of forked children
      }
    }
    return false;
  }

  //same lumi, or new lumi detected in file (old mode)
  if (!getLSFromFilename_) {
    //get new lumi from file header
    const uint32_t lumi = event_->lumi();
    if (!luminosityBlockAuxiliary()
        || luminosityBlockAuxiliary()->luminosityBlock() != lumi) {
      lastOpenedLumi_ = lumi;
      resetLuminosityBlockAuxiliary();
      timeval tv;
      gettimeofday(&tv, 0);
      edm::Timestamp lsopentime(
                                (unsigned long long) tv.tv_sec * 1000000
                                + (unsigned long long) tv.tv_usec);
      edm::LuminosityBlockAuxiliary* luminosityBlockAuxiliary =
        new edm::LuminosityBlockAuxiliary(runAuxiliary()->run(),
                                          lumi, lsopentime,
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

  eventID_ = edm::EventID(event_->run(), lastOpenedLumi_,
                          event_->event());

  setEventCached();

  return true;

}

bool FedRawDataInputSource::cacheNextEvent()
{
  if ( bufferLeft_ < (4 + 1024) * sizeof(uint32) ) //minimal size to fit any version of FRDEventHeader
  {
    if ( !readNextChunkIntoBuffer() ) return false;
  }

  event_.reset( new FRDEventMsgView(bufferCursor_) );

  const uint32_t msgSize = event_->size();

  if ( bufferLeft_ < msgSize )
  {
    if ( !readNextChunkIntoBuffer() || bufferLeft_ < msgSize )
    {
      throw cms::Exception("FedRawDataInputSource::cacheNextEvent") <<
        "Premature end of input file while reading event data";
    }
    event_.reset( new FRDEventMsgView(bufferCursor_) );
  }

  bufferLeft_ -= msgSize;
  bufferCursor_ += msgSize;

  return true;
}

bool FedRawDataInputSource::readNextChunkIntoBuffer()
{
  //this function is called when we reach the end of the buffer (i.e. bytes to read are more than bytes left in buffer)
  if (eofReached() && !openNextFile())
    return false; //should only happen when requesting the next event header and the run is over
  if (bufferLeft_ == 0) { //in the rare case the last byte barely fit
    uint32_t chunksize = 1024 * 1024 * eventChunkSize_;
    bufferLeft_ = fread((void*) dataBuffer_, sizeof(unsigned char),
                        chunksize, fileStream_); //reads a maximum of chunksize bytes from file into buffer
  } else { //refill the buffer after having moved the bufferLeft_ bytes at the head
    uint32_t chunksize = 1024 * 1024 * eventChunkSize_ - bufferLeft_;
    memcpy((void*) dataBuffer_, bufferCursor_, bufferLeft_); //this copy could be avoided
    bufferLeft_ += fread((void*) (dataBuffer_ + bufferLeft_),
                         sizeof(unsigned char), chunksize, fileStream_);
  }
  bufferCursor_ = dataBuffer_; // reset the cursor at the beginning of the buffer
  return (bufferLeft_ > 0);
}

bool FedRawDataInputSource::eofReached() const
{
  if (fileStream_ == 0)
    return true;

  int c;
  c = fgetc(fileStream_);
  ungetc(c, fileStream_);

  return (c == EOF);
}

bool FedRawDataInputSource::openNextFile()
{
  while (!searchForNextFile() && !eorFileSeen_) {
    edm::LogInfo("FedRawDataInputSource") << "No file for me... sleep and try again..." << std::endl;
    usleep(100000);
  }

  return (fileStream_ != 0 || !eorFileSeen_);
}

void FedRawDataInputSource::read(edm::EventPrincipal& eventPrincipal) 
{
  currentInputEventCount_--;
  std::auto_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  edm::Timestamp tstamp = fillFEDRawDataCollection(rawData);
  
  edm::EventAuxiliary aux(eventID_, processGUID(), tstamp, true,
                          edm::EventAuxiliary::PhysicsTrigger);
  makeEvent(eventPrincipal, aux);
  
  edm::WrapperOwningHolder edp(new edm::Wrapper<FEDRawDataCollection>(rawData),
                               edm::Wrapper<FEDRawDataCollection>::getInterface());
  
  eventPrincipal.put(daqProvenanceHelper_.constBranchDescription_, edp,
                     daqProvenanceHelper_.dummyProvenance_);
  
  return;
}

edm::Timestamp FedRawDataInputSource::fillFEDRawDataCollection(std::auto_ptr<FEDRawDataCollection>& rawData) const
{
  edm::Timestamp tstamp;
  uint32_t eventSize = event_->eventSize();
  char* event = (char*)event_->payload();
  
  while (eventSize > 0) {
    eventSize -= sizeof(fedt_t);
    const fedt_t* fedTrailer = (fedt_t*) (event + eventSize);
    const uint32_t fedSize = FED_EVSZ_EXTRACT(fedTrailer->eventsize) << 3; //trailer length counts in 8 bytes
    eventSize -= (fedSize - sizeof(fedh_t));
    const fedh_t* fedHeader = (fedh_t *) (event + eventSize);
    const uint16_t fedId = FED_SOID_EXTRACT(fedHeader->sourceid);
    if (fedId == FEDNumbering::MINTriggerGTPFEDID) {
      evf::evtn::evm_board_setformat(fedSize);
      const uint64_t gpsl = evf::evtn::getgpslow((unsigned char*) fedHeader);
      const uint64_t gpsh = evf::evtn::getgpshigh((unsigned char*) fedHeader);
      tstamp = edm::Timestamp(static_cast<edm::TimeValue_t> ((gpsh << 32) + gpsl));
    }
    FEDRawData& fedData = rawData->FEDData(fedId);
    fedData.resize(fedSize);
    memcpy(fedData.data(), event + eventSize, fedSize);
  }
  assert(eventSize == 0);

  return tstamp;
}

bool FedRawDataInputSource::searchForNextFile()
{
  if(currentInputEventCount_!=0){
    throw cms::Exception("RuntimeError") << "Went to search for next file but according to BU more events in " 
                                         << currentInputJson_.string();
  }
  boost::filesystem::remove(currentInputJson_); //the content of this file is now accounted for in the output json
  
  std::string nextFile;
  uint32_t ls;

  edm::LogInfo("FedRawDataInputSource") << "Asking for next file... to the DaqDirector";
  evf::FastMonitoringService*fms = (evf::FastMonitoringService *) (edm::Service<evf::MicroStateService>().operator->());
  fms->startedLookingForFile();
  bool fileIsOKToGrab = edm::Service<evf::EvFDaqDirector>()->updateFuLock(ls,nextFile,eorFileSeen_);

  if (fileIsOKToGrab) {

    edm::LogInfo("FedRawDataInputSource") << "The director says to grab: " << nextFile;

    fms->stoppedLookingForFile();
    edm::LogInfo("FedRawDataInputSource") << "grabbing next file, setting last seen lumi to LS = " << ls;

    boost::filesystem::path jsonFile(nextFile);
    jsonFile.replace_extension(".jsn");
    assert( grabNextJsonFile(jsonFile) );
    openDataFile(nextFile);

    if (getLSFromFilename_) lastOpenedLumi_ = ls;
    return true;
      
  } else if (ls > lastOpenedLumi_) {
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
    return false;
  }
}

bool FedRawDataInputSource::grabNextJsonFile(boost::filesystem::path const& jsonSourcePath)
{
  try {
    // assemble json destination path
    boost::filesystem::path jsonDestPath(fuOutputDir_);

    std::ostringstream fileNameWithPID;
    fileNameWithPID << jsonSourcePath.stem().string() << "_pid"
                    << std::setfill('0') << std::setw(5) << getpid() << ".jsn";
    const boost::filesystem::path filePathWithPID(fileNameWithPID.str());
    jsonDestPath /= fileNameWithPID.str();

    edm::LogInfo("FedRawDataInputSource") << " JSON rename " << jsonSourcePath << " to "
                                          << jsonDestPath;

    if ( testModeNoBuilderUnit_ )
      boost::filesystem::copy(jsonSourcePath,jsonDestPath);
    else
      boost::filesystem::rename(jsonSourcePath,jsonDestPath);

    currentInputJson_ = jsonDestPath; // store location for later deletion.
    boost::filesystem::ifstream ij(jsonDestPath);
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
    
    return true;
  }

  catch (const boost::filesystem::filesystem_error& ex)
  {
    // Input dir gone?
    edm::LogError("FedRawDataInputSource") << " - grabNextFile BOOST FILESYSTEM ERROR CAUGHT: " << ex.what()
                  << " - Maybe the BU run dir disappeared? Ending process with code 0...";
    _exit(0);
  }
  catch (std::runtime_error e)
  {
    // Another process grabbed the file and NFS did not register this
     edm::LogError("FedRawDataInputSource") << " - grabNextFile runtime Exception: " << e.what() << std::endl;
  }
  catch (std::exception e)
  {
    // BU run directory disappeared?
    edm::LogError("FedRawDataInputSource") << " - grabNextFileSOME OTHER EXCEPTION OCCURED!!!! ->" << e.what()
                                           << std::endl;
  }
  return false;
}

void FedRawDataInputSource::openDataFile(std::string const& nextFile)
{
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
    edm::LogInfo("FedRawDataInputSource") << " opened file " << nextFile;
  }
  else
  {
    throw cms::Exception("FedRawDataInputSource::openDataFile") <<
      " failed to open file " << nextFile << " fd:" << fileDescriptor;
  }
}

void FedRawDataInputSource::renameToNextFree() const
{
  boost::filesystem::path source(openFile_);
  boost::filesystem::path destination( edm::Service<evf::EvFDaqDirector>()->getJumpFilePath() );

  edm::LogInfo("FedRawDataInputSource") << "Instead of delete, RENAME: " << openFile_
                                        << " to: " << destination.string();
  boost::filesystem::rename(source,destination);
  boost::filesystem::rename(source.replace_extension(".jsn"),destination.replace_extension(".jsn"));
}

void FedRawDataInputSource::preForkReleaseResources()
{}

void FedRawDataInputSource::postForkReacquireResources(boost::shared_ptr<edm::multicore::MessageReceiverForSource>)
{
  InputSource::rewind();
  setRunAuxiliary(
                  new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(),
					edm::Timestamp::invalidTimestamp()));
}

void FedRawDataInputSource::rewind_()
{}

// define this class as an input source
DEFINE_FWK_INPUT_SOURCE( FedRawDataInputSource);

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
