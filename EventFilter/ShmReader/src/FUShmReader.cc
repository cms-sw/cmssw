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
#include "FWCore/PluginManager/interface/ModuleDef.h"


#include <iostream>


using namespace std;
using namespace evf;
using namespace edm;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FUShmReader::FUShmReader()
  : event_(0)
  , shmBuffer_(0)
  , runNumber_(0xffffffff)
  , evtNumber_(0xffffffff)
  , lastCellIndex_(0xffffffff)
{
  shmBuffer_=FUShmBuffer::getShmBuffer();
}


//______________________________________________________________________________
FUShmReader::~FUShmReader()
{
  if (0!=shmBuffer_) {
    edm::LogInfo("FUShmReader")<<"detach from shared memory segment."<<endl;
    shmdt(shmBuffer_);
  }
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
    shmBuffer_->scheduleRawCellForDiscard(lastCellIndex_);
    event_ = 0;
  }
  
  // wait for an event to become available, retrieve it
  FUShmRawCell* newCell=shmBuffer_->rawCellToRead();
  
  // if the event is 'empty', the reader is being told to shut down!
  evt::State_t state=shmBuffer_->evtState(newCell->index());
  if (state==evt::EMPTY) {
    edm::LogInfo("ShutDown")<<"Received empty event, shut down."<<endl;
    shmBuffer_->scheduleRawEmptyCellForDiscard(newCell);
    shmdt(shmBuffer_);
    shmBuffer_=0;
    event_=0;
    lastCellIndex_=0xffffffff;
    return false;
  }
  else assert(state==evt::RAWREADING);
  
  // read the event data into the fwk raw data format
  evtNumber_    =newCell->evtNumber();
  lastCellIndex_=newCell->index();
  event_        =new FEDRawDataCollection();
  for (unsigned int i=0;i<newCell->nFed();i++) {
    unsigned int fedSize=newCell->fedSize(i);
    if (fedSize>0) {
      FEDRawData& fedData=event_->FEDData(i);
      fedData.resize(fedSize);
      newCell->readFed(i,fedData.data());
    }
  }
  
  // reading the cell is finished (new state will be 'isProcessing')
  shmBuffer_->finishReadingRawCell(newCell);
  eID=EventID(runNumber_,evtNumber_);
  data=event_;
  
  return true;
}


////////////////////////////////////////////////////////////////////////////////
// CMSSW framwork macros
////////////////////////////////////////////////////////////////////////////////

DEFINE_SEAL_MODULE();
DEFINE_EDM_PLUGIN(DaqReaderPluginFactoryU,FUShmReader,"FUShmReader");
