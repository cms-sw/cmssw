////////////////////////////////////////////////////////////////////////////////
//
// RawEventSourceFromBU
// -----------------
//
//           EM - data da destinarsi
////////////////////////////////////////////////////////////////////////////////
#ifdef RAW_EVENT_SOURCE_FROM_BU_IS_READY

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Sources/interface/ExternalInputSource.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"

#include "EvFDaqDirector.h"

#include <unistd.h>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>

#include "EventFilter/FEDInterface/interface/fed_header.h"
#include "EventFilter/FEDInterface/interface/fed_trailer.h"


namespace errorstreamsource{
  static unsigned int gtpEvmId_ =  FEDNumbering::MINTriggerGTPFEDID;
  //static unsigned int gtpeId_ =  FEDNumbering::MINTriggerEGTPFEDID; // unused
}


class RawEventSourceFromBU : public edm::ExternalInputSource
{
public:
  // construction/destruction
  RawEventSourceFromBU(edm::ParameterSet const& pset,
		    edm::InputSourceDescription const& desc);
  virtual ~RawEventSourceFromBU();
  
private:
  // member functions
  void setRunAndEventInfo();
  bool produce(edm::Event& e);
  
  void beginJob();
  void beginRun(edm::Run& r){
    std::cout << "RawEventSourceFromBU: begin run " << std::endl;
    //    openFile();
  }
  void endRun(edm::Run& r) {;} 
  void beginLuminosityBlock(edm::LuminosityBlock& lb) {;}
  void endLuminosityBlock(edm::LuminosityBlock& lb) {;}
  
  bool openFile();
  
  
private:
  // member data
  std::vector<std::string>::const_iterator itFileName_;
  std::string baseDir_;
  std::string runDir_;
  unsigned int fileLumi_;
  std::ifstream fin_;
};


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
RawEventSourceFromBU::RawEventSourceFromBU(edm::ParameterSet const& pset,
				     edm::InputSourceDescription const& desc)
  : ExternalInputSource(pset,desc)
  , fileLumi_(0)
{
  produces<FEDRawDataCollection>();
}


//______________________________________________________________________________
RawEventSourceFromBU::~RawEventSourceFromBU()
{

}


//////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void RawEventSourceFromBU::beginJob()
{
  std::cout << "RawEventSourceFromBU beginjob " << std::endl;
  baseDir_ = edm::Service<evf::EvFDaqDirector>()->baseDir(); 
  runDir_ =  edm::Service<evf::EvFDaqDirector>()->findHighestRunDir();
  setRunNumber(edm::Service<evf::EvFDaqDirector>()->findHighestRun());
}


//______________________________________________________________________________
void RawEventSourceFromBU::setRunAndEventInfo()
{
  //  std::cout << "Set run and event info " << std::endl;
  uint32_t version(1);
  uint32_t runNumber(0);
  uint32_t lumiNumber(1);
  uint32_t evtNumber(0);
  bool status;
  while(!fin_.is_open()){
    if(!openFile()){ 
      setEventNumber(0);
      return;
    }
  }

  status = fin_.read((char*)&runNumber,sizeof(uint32_t));
  if (runNumber < 32) {
      version = runNumber;
      status = fin_.read((char*)&runNumber,sizeof(uint32_t));
  }
  if (version >= 2) {
      status = fin_.read((char*)&lumiNumber,sizeof(uint32_t));
  }
  status = fin_.read((char*)&evtNumber,sizeof(uint32_t));
  
  if (!status) {
    fin_.close();
    fin_.clear();
    if(fileLumi_>0)
      edm::Service<evf::EvFDaqDirector>()->removeFile(fileLumi_);
    return setRunAndEventInfo();
  }
  if(runNumber != run()) throw cms::Exception("LogicError")<< "Run number from file - " << runNumber
							   << " does not correspond to the one from directory " << run()
							   << "\n";
  if(lumiNumber != (fileLumi_)) throw cms::Exception("LogicError")<< "Ls number from readin file - " << lumiNumber 
							   << " does not correspond to the one from directory " << fileLumi_-1
							   << "\n";
  setLuminosityBlockNumber_t(lumiNumber);
  setEventNumber(evtNumber);
}


//______________________________________________________________________________
bool RawEventSourceFromBU::produce(edm::Event& e)
{
  //  std::cout << "produce " << std::endl;
  unsigned int totalEventSize = 0;
  if (!fin_.is_open()) return false;
  
  auto_ptr<FEDRawDataCollection> result(new FEDRawDataCollection());
  
  uint32_t fedSize[1024];
  fin_.read((char*)fedSize,1024*sizeof(uint32_t));
  for (unsigned int i=0;i<1024;i++) {
    //    std::cout << "fedsize " << i << " " << fedSize[i] << std::endl;
    totalEventSize += fedSize[i];
  }
  unsigned int gtpevmsize = fedSize[errorstreamsource::gtpEvmId_];
  if(gtpevmsize>0)
    evf::evtn::evm_board_setformat(gtpevmsize);
  //  std::cout << "total event size " << totalEventSize << std::endl;
  char *event = new char[totalEventSize];
  fin_.read(event,totalEventSize);
  while(totalEventSize>0) {
    totalEventSize -= 8;
    fedt_t *fedt = (fedt_t*)(event+totalEventSize);
    unsigned int fedsize = FED_EVSZ_EXTRACT(fedt->eventsize);
    fedsize *= 8; // fed size in bytes
    //    std::cout << "fedsize = " << fedsize << std::endl;
    totalEventSize -= (fedsize - 8);
    fedh_t *fedh = (fedh_t *)(event+totalEventSize);
    unsigned int soid = FED_SOID_EXTRACT(fedh->sourceid);
    if(soid==errorstreamsource::gtpEvmId_){
      unsigned int gpsl = evf::evtn::getgpslow((unsigned char*)fedh);
      unsigned int gpsh = evf::evtn::getgpshigh((unsigned char*)fedh);
      edm::TimeValue_t time = gpsh;
      time = (time << 32) + gpsl;
      setTime(time);
    }
    FEDRawData& fedData=result->FEDData(soid);
    fedData.resize(fedsize);
    memcpy(fedData.data(),event+totalEventSize,fedsize);
    //    std::cout << "now totalEventSize = " << totalEventSize << std::endl;
  }
  e.put(result);
  delete[] event;
  return true;
}


//______________________________________________________________________________
bool RawEventSourceFromBU::openFile()
{
  // get next file to open
  int lockstate = 0;


  if(lockstate < 0) return false;
  // potentially, next file is for reading so let's first check and get rid of the previous used file
  if (fin_.is_open())
    {
      fin_.close();
      fin_.clear();
      // if previous file was processed ok ask the director to delete it
      if(fileLumi_>0)
	edm::Service<evf::EvFDaqDirector>()->removeFile(fileLumi_);
    }
  // check if someone is working on the next ls, in that case, bump the ls and try again - if successful will return the right number 
  unsigned int nextLs = fileLumi_+1;
  lockstate=edm::Service<evf::EvFDaqDirector>()->updateFuLock(nextLs);
  fileLumi_ = nextLs;
  
  //check the lock is still healthy
  if(lockstate < 0) return false;

  // check if we are not hitting the current ls file being written by BU
  while((lockstate=edm::Service<evf::EvFDaqDirector>()->readBuLock())>=(int)fileLumi_) 
    { 
      ::sleep(1); 
      std::cout << "waiting for file to be released by bu for ls " << fileLumi_ << std::endl;
    }

  // get the correct filename from the director
  std::string filename = edm::Service<evf::EvFDaqDirector>()->getFileForLumi(fileLumi_+1);
  //  std::cout << " here " << fileLumi_ << std::endl;

  fin_.open(filename);
  //  fileLumi_++;
  //  std::cout << " here " << fileLumi_ << std::endl;
  std::cout << " is open " << fin_.is_open() << std::endl;
  return true;
}
				 
				 
///////////////////////////////////////////////////
// define this class as an input source
////////////////////////////////////////////////////////////////////////////////
DEFINE_FWK_INPUT_SOURCE(RawEventSourceFromBU);

#endif
