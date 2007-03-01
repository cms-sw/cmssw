////////////////////////////////////////////////////////////////////////////////
//
// FUShmReader
// -----------
//
//            11/03/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmReader/interface/FUShmReader.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "IORawData/DaqSource/interface/DaqReaderPluginFactory.h"
#include "PluginManager/ModuleDef.h"


#include <iostream>


using namespace std;
using namespace evf;
using namespace edm;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmReader::FUShmReader(const edm::ParameterSet& pset)
  : event_(0)
  , shmBuffer_(0)
  , runNumber_(0xffffffff)
  , evtNumber_(0xffffffff)
  , fuResourceId_(0xffffffff)
{
  shmBuffer_=FUShmBuffer::getShmBuffer();
}


//______________________________________________________________________________
FUShmReader::~FUShmReader()
{
  
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
bool FUShmReader::fillRawData(EventID& eID,
			      Timestamp& tstamp, 
			      FEDRawDataCollection*& data)
{
  // just in case the reader hasn't yet attached to the shm segment
  if (0==shmBuffer_) {
    shmBuffer_=FUShmBuffer::getShmBuffer();
    if(0==shmBuffer_) {
      edm::LogError("NoShmBuffer")<<"Failed to retrieve shm segment."<<endl;
      throw cms::Exception("NullPointer")<<"Failed to retrieve shm segment."<<endl;
    }
  }
  
  // discard old event
  if(0!=event_) {
    FUShmBufferCell* oldCell=shmBuffer_->cell(fuResourceId_);
    assert(oldCell->isProcessing());
    oldCell->setStateProcessed();
    shmBuffer_->scheduleForDiscard(oldCell);
    event_ = 0;
  }

  // wait for an event to become available, retrieve it
  shmBuffer_->waitReaderSem();
  FUShmBufferCell* newCell=shmBuffer_->currentReaderCell();
  
  // if the event is 'empty', the reader is being told to shut down!
  if (newCell->isEmpty()) {
    edm::LogInfo("ShutDown")<<"Received empty event, shut down."<<endl;
    shmBuffer_->postWriterSem();
    FUShmBuffer::shm_dettach((void*)shmBuffer_);
    shmBuffer_=0;
    return false;
  }
  else assert(newCell->isWritten());
  
  // read the event data into the fwk raw data format
  evtNumber_   =newCell->evtNumber();
  fuResourceId_=newCell->fuResourceId();
  event_       =new FEDRawDataCollection();
  for (unsigned int i=0;i<newCell->nFed();i++) {
    unsigned int fedSize=newCell->fedSize(i);
    if (fedSize>0) {
      FEDRawData& fedData=event_->FEDData(i);
      fedData.resize(fedSize);
      newCell->readFed(i,fedData.data());
    }
  }
  
  // set cell state to 'processing' and hand back over to the processor
  newCell->setStateProcessing();
  eID=EventID(runNumber_,evtNumber_);
  data=event_;

  return true;
}


////////////////////////////////////////////////////////////////////////////////
// CMSSW framwork macros
////////////////////////////////////////////////////////////////////////////////

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN (DaqReaderPluginFactory,FUShmReader,"FUShmReader");
