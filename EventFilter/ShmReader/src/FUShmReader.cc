////////////////////////////////////////////////////////////////////////////////
//
// FUShmReader
// -----------
//
//            11/03/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/ShmReader/interface/FUShmReader.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "IORawData/DaqSource/interface/DaqReaderPluginFactory.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"


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
  //  shmBuffer_=FUShmBuffer::getShmBuffer();
}


//______________________________________________________________________________
FUShmReader::~FUShmReader()
{
  if (0!=shmBuffer_) {
    edm::LogInfo("FUShmReader")<<"detach from shared memory segment."<<endl;
    if (lastCellIndex_<0xffffffff) {
      shmBuffer_->writeErrorEventData(runNumber_,getpid(),lastCellIndex_);
      shmBuffer_->removeClientPrcId(getpid());
    }
    shmdt(shmBuffer_);
  }
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
int FUShmReader::fillRawData(EventID& eID,
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
  
  // if the event is 'stop', the reader is being told to shut down!
  evt::State_t state=shmBuffer_->evtState(newCell->index());
  if (state==evt::STOP) {
    edm::LogInfo("ShutDown")<<"Received STOP event, shut down."<<endl;
    std::cout << getpid() << " Received STOP event, shut down." << std::endl;
    if(newCell->getEventType() != evt::STOPPER){
      edm::LogError("InconsistentLsCell") << getpid() 
					  << " GOT what claims to be a STOPPER event but eventType is " 
					  <<  newCell->getEventType()
					  << std::endl;
    }
    shmBuffer_->scheduleRawEmptyCellForDiscard(newCell);
    shmdt(shmBuffer_);
    shmBuffer_=0;
    event_=0;
    lastCellIndex_=0xffffffff;
    return 0;
  }
  else if(state==evt::LUMISECTION){
    unsigned int ls = newCell->getLumiSection();
    if(newCell->getEventType() != evt::EOL){
      edm::LogError("InconsistentLsCell") << getpid() 
					  << " GOT what claims to be an EOL event but eventType is " 
					  <<  newCell->getEventType()
					  << " and ls is " << ls
					  << std::endl;
      shmBuffer_->sem_print();
    }
    shmBuffer_->scheduleRawCellForDiscard(newCell->index());
    //write process ID for raw cell to shm
    shmBuffer_->setEvtPrcId(newCell->index(),getpid());
    if(ls==0){
      edm::LogError("ZeroLsCell") << getpid() 
				  << " GOT an EOL event for ls 0!!!" 
				  << std::endl;
      return fillRawData(eID, tstamp, data);
    }
    return (-1)*ls;
  }
  // getting an 'empty' event here is a pathological condition !!!
  else if(state==evt::EMPTY){
    edm::LogError("EmptyRawCell")
      <<"Received empty event, this should not happen !!!" <<endl;
    std::cout << getpid() << "Received EPTY event!!! ERROR." << std::endl;
    return fillRawData(eID, tstamp, data);
  }
  else assert(state==evt::RAWREADING);
  
  // read the event data into the fwk raw data format
  evtNumber_    =newCell->evtNumber();
  lastCellIndex_=newCell->index();
  //write process ID for the current raw cell to shm
  shmBuffer_->setEvtPrcId(lastCellIndex_,getpid());
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
  eID=EventID(runNumber_,1U,evtNumber_);
  data=event_;
  if(evtNumber_==0) 
    std::cout << getpid() << " ShmReader got event number zero !!! " 
	      << std::endl;
  return 1;
}


////////////////////////////////////////////////////////////////////////////////
// CMSSW framwork macros
////////////////////////////////////////////////////////////////////////////////


DEFINE_EDM_PLUGIN(DaqReaderPluginFactoryU,FUShmReader,"FUShmReader");
