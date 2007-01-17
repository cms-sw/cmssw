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
  if (0==shmBuffer_) {
    shmBuffer_=FUShmBuffer::getShmBuffer();
    if(0==shmBuffer_) {
      edm::LogError("NoShmBuffer")<<"Failed to retrieve shm segment."<<endl;
      throw cms::Exception("NullPointer")<<"Failed to retrieve shm segment."<<endl;
    }
  }
  
  if(0!=event_) {
    FUShmBufferCell* oldCell=shmBuffer_->cell(fuResourceId_);
    assert(oldCell->isRead());
    oldCell->setStateProcessed();
    shmBuffer_->scheduleForDiscard(oldCell->buResourceId());
    shmBuffer_->postWriterSem();
  }

  shmBuffer_->waitReaderSem();

  shmBuffer_->lock();
  
  FUShmBufferCell* newCell=shmBuffer_->currentReaderCell();
  assert(newCell->isWritten());
  
  shmBuffer_->unlock();

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
  
  newCell->setStateRead();
  eID=EventID(runNumber_,evtNumber_);
  data=event_;

  return true;
}


////////////////////////////////////////////////////////////////////////////////
// CMSSW framwork macros
////////////////////////////////////////////////////////////////////////////////

DEFINE_SEAL_MODULE();
DEFINE_SEAL_PLUGIN (DaqReaderPluginFactory,FUShmReader,"FUShmReader");
