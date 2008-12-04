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

#include "FWCore/Sources/interface/ExternalInputSource.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include <unistd.h>
#include <string>
#include <vector>
#include <fstream>


class ErrorStreamSource : public edm::ExternalInputSource
{
public:
  // construction/destruction
  ErrorStreamSource(edm::ParameterSet const& pset,
		    edm::InputSourceDescription const& desc);
  virtual ~ErrorStreamSource();
  
private:
  // member functions
  void setRunAndEventInfo();
  bool produce(edm::Event& e);
  
  void beginRun(edm::Run& r) {;}
  void endRun(edm::Run& r) {;} 
  void beginLuminosityBlock(edm::LuminosityBlock& lb) {;}
  void endLuminosityBlock(edm::LuminosityBlock& lb) {;}
  
  bool openFile(const std::string& fileName);
  
  
private:
  // member data
  std::vector<std::string>::const_iterator itFileName_;
  std::ifstream fin_;
};


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
ErrorStreamSource::ErrorStreamSource(edm::ParameterSet const& pset,
				     edm::InputSourceDescription const& desc)
  : ExternalInputSource(pset,desc)
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
void ErrorStreamSource::setRunAndEventInfo()
{
  uint32_t runNumber(0);
  uint32_t evtNumber(0);
  bool status;
  status = fin_.read((char*)&runNumber,sizeof(uint32_t));
  status = fin_.read((char*)&evtNumber,sizeof(uint32_t));
  
  if (!status) {
    itFileName_++; if (itFileName_==fileNames().end()) { fin_.close(); return; }
    openFile(*itFileName_);
    status = fin_.read((char*)&runNumber,sizeof(uint32_t));
    status = fin_.read((char*)&evtNumber,sizeof(uint32_t));
    if (!status) { fin_.close(); return; }
  }
  
  runNumber = (runNumber==0) ? 1 : runNumber;
  
  setRunNumber(runNumber);
  setEventNumber(evtNumber);
}


//______________________________________________________________________________
bool ErrorStreamSource::produce(edm::Event& e)
{
  if (!fin_.is_open()) return false;
  
  auto_ptr<FEDRawDataCollection> result(new FEDRawDataCollection());
  
  uint32_t fedSize[1024];
  fin_.read((char*)fedSize,1024*sizeof(uint32_t));
  for (unsigned int i=0;i<1024;i++) {
    if (fedSize[i]==0) continue;
    FEDRawData& fedData=result->FEDData(i);
    fedData.resize(fedSize[i]);
    fin_.read((char*)fedData.data(),fedSize[i]);
  }
  e.put(result);
  return true;
}


//______________________________________________________________________________
bool ErrorStreamSource::openFile(const string& fileName)
{
  fin_.close();
  fin_.clear();
  unsigned int pos = fileName.find(':');
  if (pos!=string::npos) {
    string prefix = fileName.substr(0,pos);
    if (prefix!="file") return false;
    pos++;
  }
  else pos=0;

  fin_.open(fileName.substr(pos).c_str(),ios::in|ios::binary);
  return fin_.is_open();
}
				 
				 
///////////////////////////////////////////////////
// define this class as an input source
////////////////////////////////////////////////////////////////////////////////
DEFINE_FWK_INPUT_SOURCE(ErrorStreamSource);
