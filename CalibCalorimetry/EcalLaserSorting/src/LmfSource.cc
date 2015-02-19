/*
 *  \author Philippe Gras CEA/Saclay
 */

#include <iostream>
#include <iomanip>
#include "CalibCalorimetry/EcalLaserSorting/interface/LmfSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

using namespace edm;
using namespace std;

const unsigned char LmfSource::minDataFormatVersion_ = 4;
const unsigned char LmfSource::maxDataFormatVersion_ = 5;
const unsigned LmfSource::fileHeaderSize = 2;


LmfSource::LmfSource(const ParameterSet& pset,
		     const InputSourceDescription& desc) :
  ProducerSourceBase(pset, desc, true),
  fileNames_(pset.getParameter<vector<string> >("fileNames")),
  iFile_ (-1),
  fedId_(-1),
  fileHeader_(fileHeaderSize),
  dataFormatVers_(5),
  rcRead_(false),
  preScale_(pset.getParameter<unsigned>("preScale")),
  iEvent_(0),
  iEventInFile_(0),
  indexTablePos_(0),
  orderedRead_(pset.getParameter<bool>("orderedRead")),
  watchFileList_(pset.getParameter<bool>("watchFileList")),
  fileListName_(pset.getParameter<std::string>("fileListName")),
  inputDir_(pset.getParameter<std::string>("inputDir")),
  nSecondsToSleep_(pset.getParameter<int>("nSecondsToSleep")),
  verbosity_(pset.getUntrackedParameter<int>("verbosity"))
{
  if(preScale_==0) preScale_ = 1;
  produces<FEDRawDataCollection>();
  // open fileListName
  if (watchFileList_) {
    fileList_.open( fileListName_.c_str() );
    if (fileList_.fail()) {
      throw cms::Exception("FileListOpenError")
        << "Failed to open input file " << fileListName_ << "\n";
    }
  } else {
  //throws a cms exception if error in fileNames parameter
  checkFileNames();
  }
}

bool LmfSource::readFileHeader(){
  if(iFile_==-1) return false; //no file open

  if(verbosity_) cout << "[LmfSource]"
                   << "Opening file #" << (iFile_+1) << " '"
                   << currentFileName_ << "'\n";
  
  in_.read((char*)&fileHeader_[0], fileHeaderSize*sizeof(uint32_t));

  if(in_.eof()) return false;
  
  if(verbosity_){
    cout << "[LmfSource]"
         << "File header (in hex):" << hex;
    for(unsigned i=0; i < fileHeaderSize; ++i){ 
      if(i%8==0) cout << "\n";
      cout << setw(8) << fileHeader_[i] << " ";
    }
    cout << dec << "\n";
  }


  char id[4];

  id[0] = fileHeader_[0] & 0xFF;
  id[1] = (fileHeader_[0] >>8) & 0xFF;
  id[2] = (fileHeader_[0] >>16) & 0xFF;
  id[3] = (fileHeader_[0] >>24) & 0xFF;
  
  if(!(id[0]=='L' && id[1] == 'M'
       && id[2] == 'F')){
    throw cms::Exception("FileReadError")
      << currentFileName_ << " is not a file in LMF format!";
  }
  dataFormatVers_ = id[3];
  if(verbosity_) cout << "[LmfSource]"
                   << "LMF format: " << (int)dataFormatVers_ << "\n";
  
  if(dataFormatVers_ > maxDataFormatVersion_
     || dataFormatVers_ < minDataFormatVersion_){
    throw cms::Exception("FileReadError")
      << currentFileName_ << ": LMF format version " << (int) dataFormatVers_
      << " is not supported by this release of LmfSource module";
  }

  indexTablePos_ = fileHeader_[1];

  if(verbosity_) cout << "[LmfSource] File position of index table: 0x"
                   << setfill('0') << hex << setw(8) << indexTablePos_
                   << setfill(' ') << dec << "\n";
  
  if(dataFormatVers_ < 5){
    in_.ignore(4);
  }
    
  return true;
}

void LmfSource::produce(edm::Event& evt){
  //   bool rc;
  //   while(!((rc = readFileHeader()) & readEventPayload())){
  //     if(openFile(++iFile_)==false){//no more files
  //       if(verbosity_) cout << "[LmfSource]"
  // 		       << "No more input file";
  //       return false;
  //     }
  //   }
  auto_ptr<FEDRawDataCollection> coll(new FEDRawDataCollection);
  coll->swap(fedColl_);
  if(verbosity_) cout << "[LmfSource] Putting FEDRawDataCollection in event\n";
  evt.put(coll);
}

bool LmfSource::openFile(int iFile){
  iEventInFile_ = 0;
  if(watchFileList_) {
    for ( ;; ) {
      // read the first field of the line, which must be the filename
      fileList_ >> currentFileName_;
      currentFileName_ = inputDir_ + "/" + currentFileName_;
      if (!fileList_.fail()) {
        // skip the rest of the line
        std::string tmp_buffer;
        std::getline(fileList_, tmp_buffer);
        if(verbosity_) cout << "[LmfSource]"
          << "Opening file " << currentFileName_ << "\n";
        in_.open(currentFileName_.c_str());
        if (!in_.fail()) {
          // file was successfully open
          return true;
        } else {
          // skip file
          edm::LogError("FileOpenError")
            << "Failed to open input file " << currentFileName_ << ". Skipping file\n";
          in_.close();
          in_.clear();
        }
      }
      // if here, no new file is available: sleep and retry later
      if (verbosity_) std::cout << "[LmfSource]"
        << " going to sleep 5 seconds\n";
      sleep(nSecondsToSleep_);
      fileList_.clear();
    }
  } else {
    if(iFile > (int)fileNames_.size()-1) return false;
    currentFileName_ = fileNames_[iFile];
    if(verbosity_) cout << "[LmfSource]"
      << "Opening file " << currentFileName_ << "\n";
    in_.open(currentFileName_.c_str());
    if(in_.fail()){
      throw cms::Exception("FileOpenError")
        << "Failed to open input file " << currentFileName_ << "\n";
    }
  }
  return true;
}

bool LmfSource::nextEventWithinFile(){
  if(iFile_<0) return false; //no file opened.
  if(orderedRead_){
    if(iEventInFile_>=indexTable_.size()) return false;
    if(verbosity_){
      cout << "[LmfSource] move to event with orbit Id "
           << indexTable_[iEventInFile_].orbit
           << " at file position 0x"
           << hex << setfill('0')
           << setw(8) << indexTable_[iEventInFile_].filePos
           << setfill(' ') << dec << "\n";
    }
    const streampos pos = indexTable_[iEventInFile_].filePos;
    in_.clear();
    in_.seekg(pos);
    if(in_.bad()){
      cout << "[LmfSource] Problem while reading file "
           << currentFileName_ << ". Problem with event index table?\n";
      return false;
    }
    ++iEventInFile_;
    return true;
  } else{
    return true;
  }
}
               
bool LmfSource::readEvent(bool doSkip){
  while(!(nextEventWithinFile() && readEventWithinFile(doSkip))){
    //failed to read event. Let's look for next file:
    in_.close();
    in_.clear();
    bool rcOpen = openFile(++iFile_);
    if(rcOpen==false){//no more files
      if(verbosity_) cout << "[LmfSource]"
		       << "No more input file";
      rcRead_ = false;
      return rcRead_;
    }
    rcRead_ = readFileHeader();
    if(verbosity_) cout << "File header readout "
		     << (rcRead_?"succeeded":"failed") << "\n";
    if(rcRead_ && orderedRead_) readIndexTable();
  }
  return rcRead_;
}
 
bool LmfSource::setRunAndEventInfo(EventID& id, TimeValue_t& time, edm::EventAuxiliary::ExperimentType& eType){
  //empties collection:
  if(fedId_>0){
    fedColl_.FEDData(fedId_).resize(0);
  }
  if(verbosity_) cout << "[LmfSource]"
		   << "About to read event...\n";

  bool rc;
  for(;;){
    if(filter()){//event to read
      rc = readEvent();
      break;    //either event is read or no more event
    } else { //event to skip
      rc = readEvent(true);
      if(rc==false){//no more events
	break;
      }
    }
  }

  if(!rc) return false; //event readout failed
    
  if(verbosity_) cout << "[LmfSource]"
		   << "Setting event time to "
		   << /*toString(*/timeStamp_/*)*/ << ", "
		   << "Run number to " << runNum_ << ","
		   << "Event number to " << eventNum_ << "\n";

  time = timeStamp_;
  id = EventID(runNum_, lumiBlock_, eventNum_);
  return true; 
}

bool LmfSource::readEventWithinFile(bool doSkip){
  if(iFile_==-1 || !rcRead_) return false; //no file open
  //                                         or header reading failed
  //number of 32-bit word to read first to get the event size
  //field
  const int timeStamp32[]      = {0, 0}; //timestamp is 64-bit long
  const int lumiBlock32[]      = {2, 2};
  const int runNum32[]         = {3, 3};
  const int orbitNum32[]       = {4, 4};
  const int bx32[]             = {5, 5};
  const int eventNum32[]       = {6, 6};
  const int activeFedId32[]    = {7,-1};
  const int calibTrig32[]      = {-1,7};
  const int nFeds32[]          = {-1,8};
  // const int reserved32[]       = {-1,9};
  const int evtHeadSize32[]    = {8,10};
  
  const unsigned char iv = dataFormatVers_-minDataFormatVersion_;
  assert(iv<=sizeof(timeStamp32)/sizeof(timeStamp32[0]));
  
  if((int)header_.size() < evtHeadSize32[iv]) header_.resize(evtHeadSize32[iv]);


  if(verbosity_) cout << "[LmfSource]"
		   << "Reading event header\n";

  in_.read((char*)&header_[0], evtHeadSize32[iv]*4);
  if(in_.bad()){//reading error other than eof
    throw cms::Exception("FileReadError")
      << "Error while reading from file " << currentFileName_;
  }
  if(in_.eof()) return false;

  if(verbosity_){
    cout << "[LmfSource]"
	 << "Event header (in hex):" << hex << setfill('0');
    for(int i=0; i < evtHeadSize32[iv]; ++i){ 
      if(i%8==0) cout << "\n";
      cout << setw(8) << header_[i] << " ";
    }
    cout << dec << setfill(' ') << "\n";
  }
  
  timeStamp_   = *(uint64_t*)&header_[timeStamp32[iv]];
  lumiBlock_   = header_[lumiBlock32[iv]];
  runNum_      = header_[runNum32[iv]];
  orbitNum_    = header_[orbitNum32[iv]];
  eventNum_    = header_[eventNum32[iv]];
  bx_          = header_[bx32[iv]];
  calibTrig_   = calibTrig32[iv]>=0?header_[calibTrig32[iv]]:0;
  int activeFedId = activeFedId32[iv]>=0?
    header_[activeFedId32[iv]]:
    ((calibTrig_ & 0x3F) + 600);
  nFeds_       = nFeds32[iv] < 0 ? 1 : header_[nFeds32[iv]];
  
  if(verbosity_){
    time_t t = time_t(timeStamp_ >>32);
    div_t t_ms_us = div(timeStamp_ & 0xFFFFFFFF, 1000);
    char tbuf[256];
    strftime(tbuf, sizeof(tbuf), "%F %T", localtime(&t));
    tbuf[sizeof(tbuf)-1] = 0;
    cout << "[LmfSource] "
	 << "timeStamp:          " << /*toString(timeStamp_)*/ timeStamp_ 
	 << " (" << tbuf << " " << t_ms_us.quot << " ms " << t_ms_us.rem   << " us)\n"
	 << "lumiBlock:          " << lumiBlock_ << "\n"
	 << "runNum:             " << runNum_ << "\n"
	 << "orbitNum:           " << orbitNum_ << "\n"
	 << "eventNum:           " << eventNum_ << "\n"
	 << "bx:                 " << bx_ << "\n"
	 << "activeFedId:        " << activeFedId << "\n"
         << "Calib trigger type: " << ((calibTrig_ >>8) & 0x3) << "\n"
         << "Color:              " << ((calibTrig_ >>6) & 0x3) << "\n"
	 << "Side:               " << ((calibTrig_ >>11) & 0x1) << "\n"
         << "nFeds:              " << nFeds_ << "\n";
  }

  const int dccLenOffset32 = 2;
  const int fedIdOffset32  = 0; 
  const int nPreRead32     = 3;
  vector<int32_t> buf(nPreRead32);
  for(int iFed = 0; iFed < nFeds_; ++iFed){  
    in_.read((char*) &buf[0], nPreRead32*sizeof(uint32_t));

    if(verbosity_){
      cout << "[LmfSource] " << nPreRead32 << " first 32-bit words of "
           << "FED block: " << hex << setfill('0');
      for(unsigned i = 0; i< buf.size(); ++i){
        cout << "0x" << setw(8) << buf[i] << " ";
      }
      cout << dec << setfill(' ');
    }
        
    
    if(in_.bad()) return false;
    
    const unsigned eventSize64 = buf[dccLenOffset32] &  0x00FFFFFF;
    const unsigned eventSize32 = eventSize64*2;
    const unsigned eventSize8  = eventSize64*8;
    const unsigned fedId_      = (buf[fedIdOffset32] >>8) & 0xFFF;

    if(eventSize8 > maxEventSize_){
      throw cms::Exception("FileReadError")
        << "Size of event fragment (FED block) read from "
        << " data of file " << currentFileName_
        << "is unexpctively large (" << (eventSize8 >>10)
        << " kByte). "
        << "This must be an error (corrupted file?)\n";
    }
    
    if(!FEDNumbering::inRange(fedId_)){
      throw cms::Exception("FileReadError")
        << "Invalid FED number read from data file.";
    }

    int32_t toRead8 = (eventSize32-nPreRead32)*sizeof(int32_t);

    if(toRead8<0){
      throw cms::Exception("FileReadError")
        << "Event size error while reading an event from file "
        << currentFileName_ << "\n";
    }
    
    if(doSkip){//event to skip
      if(verbosity_) cout << "[LmfSource] "
                       << "Skipping on event. Move file pointer "
                       << toRead8 << " ahead.\n";
      in_.seekg(toRead8, ios::cur);
      if(in_.bad()){//reading error other than eof
        throw cms::Exception("FileReadError")
          << "Error while reading from file " << currentFileName_;
      }
    } else{
      //reads FED data:
      FEDRawData& data_ = fedColl_.FEDData(fedId_);
      data_.resize(eventSize8);
      
      //copy already read data:
      copy(buf.begin(), buf.end(), (int32_t*)data_.data());
      
      in_.read((char*)(data_.data()) + nPreRead32*4,
               toRead8);
      
      if(in_.bad()){//reading error other than eof
        throw cms::Exception("FileReadError")
          << "Error while reading from file " << currentFileName_;
      }
    
      if(verbosity_ && data_.size()>16){
        cout << "[LmfSource]"
             << "Head of DCC data (in hex):" << hex;
        for(int i=0; i < 16; ++i){ 
          if(i%8==0) cout << "\n";
          cout << setw(8) << ((uint32_t*)data_.data())[i] << " ";
        }
        cout << dec << "\n";
      }

      if(dataFormatVers_<=4){//calib trigger in not in event header.
        //                    gets it from DCC block
        calibTrig_ = (((uint32_t*)data_.data())[5] & 0xFC0)
          | ((activeFedId-600) &  0x3F);
        if(verbosity_){
          cout << "[LmfSource] Old data format. "
            "Uses information read from FED block to retrieve calibration "
            "trigger type. Value is: 0x"
               << hex << setfill('0') << setw(3) << calibTrig_
               << setfill(' ') << dec << "\n";
        }
      }
    }
    if(in_.eof()) return false;
  }
  ++iEvent_;
  return true;
}

bool LmfSource::filter() const{
  return (iEvent_%preScale_==0);
}
	    
std::string LmfSource::toString(TimeValue_t& t) const{
  char buf[256];
  const int secTousec = 1000*1000;
  time_t tsec = t/secTousec;
  uint32_t tusec = (uint32_t)(t-tsec);
  strftime(buf, sizeof(buf), "%F %R %S s", localtime(&tsec));
  buf[sizeof(buf)-1] = 0;
  stringstream buf2;
  buf2 << (tusec+500)/1000;
  return string(buf) + " " + buf2.str() + " ms";
}

void LmfSource::checkFileNames(){
  for(unsigned i = 0; i < fileNames_.size(); ++i){
    std::string& fileName = fileNames_[i];
    const char s[] = "file:";
    if(fileName.compare(0, sizeof(s)-1, s)==0){ //file: prefix => to strip
      fileName.erase(fileName.begin(),
		     fileName.begin() + sizeof(s)-1);
    }
    if(fileName.find_first_of(":")!=string::npos){
      throw cms::Exception("LmfSource")
	<< "Character ':' is not allowed in paths specified fileNames "
	<< "parameter. Please note only local file (or NFS, AFS)"
	<< " is supported (no rfio, no /store)";
    }
    const char s1[] = "/store";
    if(fileName.compare(0, sizeof(s1)-1, s1)==0){
      throw cms::Exception("LmfSource")
	<< "CMSSW /store not supported by LmfSource. Only local file "
	<< "(or NFS/AFS) allowed. Path starting with /store not permitted";
    }
  }
}

void LmfSource::readIndexTable(){

  stringstream errMsg;
  errMsg << "Error while reading event index table of file "
         << currentFileName_ << ". Try to read it with "
         << "option orderedRead disabled.\n";

  if(indexTablePos_==0) throw cms::Exception("LmfSource") << errMsg.str();

  in_.clear();
  in_.seekg(indexTablePos_);

  uint32_t nevts = 0;
  in_.read((char*)&nevts, sizeof(nevts));
  in_.ignore(4);
  if(nevts>maxEvents_){
    throw cms::Exception("LmfSource")
      << "Number of events indicated in event index of file "
      << currentFileName_ << " is unexpectively large. File cannot be "
      << "read in time-ordered event mode. See orderedRead parmater of "
      << "LmfSource module.\n";
  }
  //if(in_.bad()) throw cms::Exception("LmfSource") << errMsg.str();
  if(in_.bad()) throw cms::Exception("LmfSource") << errMsg.str();
  indexTable_.resize(nevts);
  in_.read((char*)&indexTable_[0], nevts*sizeof(IndexRecord));
}
