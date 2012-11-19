////////////////////////////////////////////////////////////////////////////////
//
// ErrorStreamSource
// -----------------
//
//            05/23/2008 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Sources/interface/ProducerSourceFromFiles.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/Provenance/interface/Timestamp.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"

#include <unistd.h>
#include <string>
#include <vector>
#include <fstream>
#include <cstring>
#include "interface/shared/fed_header.h"
#include "interface/shared/fed_trailer.h"


namespace errorstreamsource{
  constexpr unsigned int gtpEvmId_ =  FEDNumbering::MINTriggerGTPFEDID;
  //static unsigned int gtpeId_ =  FEDNumbering::MINTriggerEGTPFEDID; // unused
}


class ErrorStreamSource : public edm::ProducerSourceFromFiles
{
public:
  // construction/destruction
  ErrorStreamSource(edm::ParameterSet const& pset,
		    edm::InputSourceDescription const& desc);
  virtual ~ErrorStreamSource();
  
private:
  // member functions
  virtual bool setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& theTime);
  virtual void produce(edm::Event& e);
  
  void beginRun(edm::Run& r) {;}
  void endRun(edm::Run& r) {;} 
  void beginLuminosityBlock(edm::LuminosityBlock& lb) {;}
  void endLuminosityBlock(edm::LuminosityBlock& lb) {;}
  
  bool openFile(const std::string& fileName);
  
  
private:
  // member data
  std::vector<std::string>::const_iterator itFileName_;
  std::ifstream fin_;
  std::auto_ptr<FEDRawDataCollection> result_;
};

////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
ErrorStreamSource::ErrorStreamSource(edm::ParameterSet const& pset,
				     edm::InputSourceDescription const& desc)
  : ProducerSourceFromFiles(pset,desc,true)
{
  itFileName_=fileNames().begin();
  openFile(*itFileName_);
  produces<FEDRawDataCollection>();
}


//______________________________________________________________________________
ErrorStreamSource::~ErrorStreamSource()
{

}


//////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
bool ErrorStreamSource::setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& theTime)
{
  uint32_t version(1);
  uint32_t runNumber(0);
  uint32_t lumiNumber(1);
  uint32_t evtNumber(0);
  bool status;
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
    itFileName_++; if (itFileName_==fileNames().end()) { fin_.close(); return false; }
    openFile(*itFileName_);
    status = fin_.read((char*)&runNumber,sizeof(uint32_t));
    if (runNumber < 32) {
        version = runNumber;
        status = fin_.read((char*)&runNumber,sizeof(uint32_t));
    }
    if (version >= 2) {
        status = fin_.read((char*)&lumiNumber,sizeof(uint32_t));
    }
    status = fin_.read((char*)&evtNumber,sizeof(uint32_t));
    if (!status) { fin_.close(); return false; }
  }
  
  runNumber = (runNumber==0) ? 1 : runNumber;
  
  id = edm::EventID(runNumber, lumiNumber, evtNumber);

//______________________________________________________________________________
  unsigned int totalEventSize = 0;
  
  result_.reset(new FEDRawDataCollection());
  
  uint32_t fedSize[1024];
  fin_.read((char*)fedSize,1024*sizeof(uint32_t));
  for (unsigned int i=0;i<1024;i++) {
    totalEventSize += fedSize[i];
  }
  unsigned int gtpevmsize = fedSize[errorstreamsource::gtpEvmId_];
  if(gtpevmsize>0)
    evf::evtn::evm_board_setformat(gtpevmsize);
  char *event = new char[totalEventSize];
  fin_.read(event,totalEventSize);
  while(totalEventSize>0) {
    totalEventSize -= 8;
    fedt_t *fedt = (fedt_t*)(event+totalEventSize);
    unsigned int fedsize = FED_EVSZ_EXTRACT(fedt->eventsize);
    fedsize *= 8; // fed size in bytes
    totalEventSize -= (fedsize - 8);
    fedh_t *fedh = (fedh_t *)(event+totalEventSize);
    unsigned int soid = FED_SOID_EXTRACT(fedh->sourceid);
    if(soid==errorstreamsource::gtpEvmId_){
      unsigned int gpsl = evf::evtn::getgpslow((unsigned char*)fedh);
      unsigned int gpsh = evf::evtn::getgpshigh((unsigned char*)fedh);
      edm::TimeValue_t time = gpsh;
      time = (time << 32) + gpsl;
      theTime = time;
    }
    FEDRawData& fedData=result_->FEDData(soid);
    fedData.resize(fedsize);
    memcpy(fedData.data(),event+totalEventSize,fedsize);
  }
  delete[] event;
  return true;
}

void ErrorStreamSource::produce(edm::Event& e) {
  e.put(result_);
}

//______________________________________________________________________________
bool ErrorStreamSource::openFile(const std::string& fileName)
{
  fin_.close();
  fin_.clear();
  size_t pos = fileName.find(':');
  if (pos!=std::string::npos) {
    std::string prefix = fileName.substr(0,pos);
    if (prefix!="file") return false;
    pos++;
  }
  else pos=0;

  fin_.open(fileName.substr(pos).c_str(),std::ios::in|std::ios::binary);
  return fin_.is_open();
}
				 
				 
///////////////////////////////////////////////////
// define this class as an input source
////////////////////////////////////////////////////////////////////////////////
DEFINE_FWK_INPUT_SOURCE(ErrorStreamSource);
