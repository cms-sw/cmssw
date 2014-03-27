#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/types.h>
#include <sys/file.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <zlib.h>

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
  verifyAdler32_(pset.getUntrackedParameter<bool> ("verifyAdler32", true)),
  testModeNoBuilderUnit_(edm::Service<evf::EvFDaqDirector>()->getTestModeNoBuilderUnit()),
  runNumber_(edm::Service<evf::EvFDaqDirector>()->getRunNumber()),
  buInputDir_(edm::Service<evf::EvFDaqDirector>()->buBaseDir()),
  fuOutputDir_(edm::Service<evf::EvFDaqDirector>()->fuBaseDir()),
  daqProvenanceHelper_(edm::TypeID(typeid(FEDRawDataCollection))),
  fileStream_(0),
  eventID_(),
  processHistoryID_(),
  currentLumiSection_(0),
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

  processHistoryID_ = daqProvenanceHelper_.daqInit(productRegistryUpdate(), processHistoryRegistryForUpdate());
  setNewRun();
  setRunAuxiliary(new edm::RunAuxiliary(runNumber_, edm::Timestamp::beginOfTime(),
					edm::Timestamp::invalidTimestamp()));
  runAuxiliary()->setProcessHistoryID(processHistoryID_);
}

FedRawDataInputSource::~FedRawDataInputSource()
{
  if (fileStream_)
    fclose(fileStream_);
  fileStream_ = 0;
}

bool FedRawDataInputSource::checkNextEvent()
{
  int eventAvailable = cacheNextEvent();
  if (eventAvailable < 0) {
    // run has ended
    resetLuminosityBlockAuxiliary();
    closeCurrentFile();
    return false;
  }
  else if(eventAvailable == 0) {
    edm::LogInfo("FedRawDataInputSource") << "No Event files at this time, but a new lumisection was detected : " << currentLumiSection_;
    
    return true;
  }
  else {
    if (!getLSFromFilename_) {
      //get new lumi from file header
      maybeOpenNewLumiSection( event_->lumi() );
    }

    eventID_ = edm::EventID(
                            event_->run(),
                            currentLumiSection_,
                            event_->event());
    
    setEventCached();
    
    return true;
  }
}

void FedRawDataInputSource::maybeOpenNewLumiSection(const uint32_t lumiSection)
{
  if (!luminosityBlockAuxiliary()
    || luminosityBlockAuxiliary()->luminosityBlock() != lumiSection) {

    if ( currentLumiSection_ > 0 ) {
      const string fuEoLS =
        edm::Service<evf::EvFDaqDirector>()->getEoLSFilePathOnFU(currentLumiSection_);
      struct stat buf;
      bool found = (stat(fuEoLS.c_str(), &buf) == 0);
      if ( !found ) {
        int eol_fd = open(fuEoLS.c_str(), O_RDWR|O_CREAT, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
        close(eol_fd);
      }
    }

    currentLumiSection_ = lumiSection;

    resetLuminosityBlockAuxiliary();

    timeval tv;
    gettimeofday(&tv, 0);
    const edm::Timestamp lsopentime( (unsigned long long) tv.tv_sec * 1000000 + (unsigned long long) tv.tv_usec );

    edm::LuminosityBlockAuxiliary* lumiBlockAuxiliary =
      new edm::LuminosityBlockAuxiliary(
        runAuxiliary()->run(),
        lumiSection, lsopentime,
        edm::Timestamp::invalidTimestamp());

    setLuminosityBlockAuxiliary(lumiBlockAuxiliary);
    luminosityBlockAuxiliary()->setProcessHistoryID(processHistoryID_);

    edm::LogInfo("FedRawDataInputSource") << "New lumi section " << lumiSection << " opened";
  }
}

int FedRawDataInputSource::cacheNextEvent()
{
  //return values or cachenext -1 :== run ended, 0:== LS ended, 1 :== cache good
  if ( bufferLeft_ < (4 + 1024) * sizeof(uint32) ) //minimal size to fit any version of FRDEventHeader
  {
    int check = readNextChunkIntoBuffer();
    if ( check ==-1) return  0;
    if ( check ==-100) return -1;
  }

  event_.reset( new FRDEventMsgView(bufferCursor_) );

  const uint32_t msgSize = event_->size();

  if ( bufferLeft_ < msgSize )
  {
    if ( readNextChunkIntoBuffer()<0 || bufferLeft_ < msgSize )
    {
      throw cms::Exception("FedRawDataInputSource::cacheNextEvent") <<
        "Premature end of input file while reading event data";
    }
    event_.reset( new FRDEventMsgView(bufferCursor_) );
  }

  if ( verifyAdler32_ && event_->version() >= 3 )
  {
    uint32_t adler = adler32(0L,Z_NULL,0);
    adler = adler32(adler,(Bytef*)event_->payload(),event_->eventSize());

    if ( adler != event_->adler32() ) {
      throw cms::Exception("FedRawDataInputSource::cacheNextEvent") <<
        "Found a wrong Adler32 checksum: expected 0x" << std::hex << event_->adler32() <<
        " but calculated 0x" << adler;
    }
  }

  bufferLeft_ -= msgSize;
  bufferCursor_ += msgSize;

  return 1;
}

int FedRawDataInputSource::readNextChunkIntoBuffer()
{
  //this function is called when we reach the end of the buffer (i.e. bytes to read are more than bytes left in buffer)
  // NOTA BENE: A positive or 0 value is returned if data are buffered
  // a value of -1 indicates no data can be buffered but there is a new 
  // lumi section to account for 
  int fileStatus = 100; //file is healthy for now 
  if (eofReached()){
    closeCurrentFile();
    fileStatus = openNextFile(); // this can now return even if there is 
                                 //no file only temporarily
    if(fileStatus==0) return -100; //should only happen when requesting the next event header and the run is over
  }
  if(fileStatus==100){ //either file was not over or a new one was opened
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
    return bufferLeft_;
  }
  else{
    // no file to read but a new lumi section has been cached 
    return -1;
  }
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

void FedRawDataInputSource::closeCurrentFile()
{
  if (fileStream_) {

    edm::LogInfo("FedRawDataInputSource") << "Closing input file " << openFile_.string();

    fclose(fileStream_);
    fileStream_ = 0;

    if (!testModeNoBuilderUnit_) {
      boost::filesystem::remove(openFile_); // won't work in case of forked children
    } else {
      renameToNextFree();
    }
  }
}

int FedRawDataInputSource::openNextFile()
{
  int nextfile = -1;
  while((nextfile = searchForNextFile())<0){
    if(eorFileSeen_)
      return 0;
    else{
      edm::LogInfo("FedRawDataInputSource") << "No file for me... sleep and try again..." << std::endl;
      usleep(100000);
    }
  }
  return nextfile;
}

void FedRawDataInputSource::read(edm::EventPrincipal& eventPrincipal) 
{
  currentInputEventCount_--;
  std::auto_ptr<FEDRawDataCollection> rawData(new FEDRawDataCollection);
  edm::Timestamp tstamp = fillFEDRawDataCollection(rawData);
  
  edm::EventAuxiliary aux(eventID_, processGUID(), tstamp, true,
                          edm::EventAuxiliary::PhysicsTrigger);
  aux.setProcessHistoryID(processHistoryID_);
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

int FedRawDataInputSource::searchForNextFile()
{
  int retval = -1;
  if(currentInputEventCount_!=0){
    throw cms::Exception("RuntimeError") << "Went to search for next file but according to BU more events in " 
                                         << currentInputJson_.string();
  }
  
  std::string nextFile;
  uint32_t ls;

  edm::LogInfo("FedRawDataInputSource") << "Asking for next file... to the DaqDirector";
  evf::FastMonitoringService*fms = (evf::FastMonitoringService *) (edm::Service<evf::MicroStateService>().operator->());
  fms->startedLookingForFile();
  bool fileIsOKToGrab = edm::Service<evf::EvFDaqDirector>()->updateFuLock(ls,nextFile,eorFileSeen_);

  if (fileIsOKToGrab) {

    edm::LogInfo("FedRawDataInputSource") << "The director says to grab: " << nextFile;

    fms->stoppedLookingForFile();

    boost::filesystem::path jsonFile(nextFile);
    jsonFile.replace_extension(".jsn");
    assert( grabNextJsonFile(jsonFile) );
    openDataFile(nextFile);
    retval = 100;
  } else {
    edm::LogInfo("FedRawDataInputSource") << "The DAQ Director has nothing for me! ";
  }

  while( getLSFromFilename_ && ls > currentLumiSection_ ) {
    maybeOpenNewLumiSection(ls);
    retval = 1;
  }

  return retval;
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
    else {
      //boost::filesystem::rename(jsonSourcePath,jsonDestPath);
      boost::filesystem::copy(jsonSourcePath,jsonDestPath);
      boost::filesystem::remove(jsonSourcePath);
    }

    currentInputJson_ = jsonDestPath; // store location for later deletion.
    boost::filesystem::ifstream ij(jsonDestPath);
    Json::Value deserializeRoot;
    DataPoint dp;
    if(!reader_.parse(ij,deserializeRoot)){
      throw std::runtime_error("Cannot deserialize input JSON file");
    }
    else {
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
