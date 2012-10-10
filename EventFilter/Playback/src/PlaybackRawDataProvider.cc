////////////////////////////////////////////////////////////////////////////////
//
// PlaybackRawDataProvider
// -----------------------
//
//            21/09/2006 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////


#include "EventFilter/Playback/interface/PlaybackRawDataProvider.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include <cstring>

using namespace std;
using namespace edm;

////////////////////////////////////////////////////////////////////////////////
// initialize static data members
////////////////////////////////////////////////////////////////////////////////

PlaybackRawDataProvider* PlaybackRawDataProvider::instance_=0;


////////////////////////////////////////////////////////////////////////////////
// construction/destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
PlaybackRawDataProvider::PlaybackRawDataProvider(const ParameterSet& iConfig)
  : queueSize_(0)
  , eventQueue_(0)
  , runNumber_(0)
  , evtNumber_(0)
  , count_(0)
  , writeIndex_(0)
  , readIndex_(0)
  , freeToEof_(false)
  , filesClosed_(false)
  , destroying_(false)
{
  queueSize_=iConfig.getUntrackedParameter<unsigned int>("queueSize",32);
  sem_init(&lock_,0,1);
  sem_init(&writeSem_,0,queueSize_);
  sem_init(&readSem_,0,0);
  runNumber_ =new unsigned int[queueSize_];
  evtNumber_ =new unsigned int[queueSize_];
  eventQueue_=new FEDRawDataCollection*[queueSize_];
  for (unsigned int i=0;i<queueSize_;i++) eventQueue_[i]=0;
  edm::LogInfo("PbImpl") << "Created Concrete RawData Provider 0x"<< hex << (unsigned long) this << dec << endl;
  instance_=this;
}


//______________________________________________________________________________
PlaybackRawDataProvider::~PlaybackRawDataProvider()
{
  if (0!=runNumber_) delete [] runNumber_;
  if (0!=evtNumber_) delete [] evtNumber_;
  if (0!=eventQueue_) {
    for (unsigned int i=0;i<queueSize_;i++)
      if (0!=eventQueue_[i]) delete eventQueue_[i];
    delete [] eventQueue_;
  }
  edm::LogInfo("PbImpl") << "Destroyed Concrete RawData Provider 0x"<< hex << (unsigned long) this << dec << endl;
  instance_=0;

  destroying_=true;
  postReadSem();
  postWriteSem();
  unlock(); 
  sem_destroy(&lock_);
  sem_destroy(&writeSem_);
  sem_destroy(&readSem_);
  usleep(10000);
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void PlaybackRawDataProvider::analyze(const Event& iEvent,
				      const EventSetup& iSetup)
{
  if(freeToEof_) return;
  waitWriteSem();
  if(freeToEof_) return;
  runNumber_[writeIndex_]=iEvent.id().run();
  evtNumber_[writeIndex_]=iEvent.id().event();

  Handle<FEDRawDataCollection> pRawData;
  iEvent.getByLabel("rawDataCollector", pRawData);
  
  if (!pRawData.isValid()) {
    edm::LogError("InvalidHandle")<<"no raw data found!"<<endl;
    return;
  }
  
  
  // copy the raw data collection into rawData_, retrievable via getFEDRawData()
  assert(0==eventQueue_[writeIndex_]);
  eventQueue_[writeIndex_]=new FEDRawDataCollection();
  for (unsigned int i=0;i<(unsigned int)FEDNumbering::MAXFEDID+1;i++) {
    unsigned int fedSize=pRawData->FEDData(i).size();
    if (fedSize>0) {
      FEDRawData& fedData=eventQueue_[writeIndex_]->FEDData(i);
      fedData.resize(fedSize);
      memcpy(fedData.data(),pRawData->FEDData(i).data(),fedSize);
    }
  }
  
  lock();
  writeIndex_=(writeIndex_+1)%queueSize_;
  count_++;
  unlock();
  
  postReadSem();

  return;
}


//______________________________________________________________________________
void PlaybackRawDataProvider::beginJob()
{
  
}

//______________________________________________________________________________
void PlaybackRawDataProvider::endJob()
{
  edm::LogInfo("Summary")<<count_<<" events read."<<endl;
}

//______________________________________________________________________________
void PlaybackRawDataProvider::respondToCloseInputFile(edm::FileBlock const& fb)
{
  filesClosed_ = true;
}

//______________________________________________________________________________
FEDRawDataCollection* PlaybackRawDataProvider::getFEDRawData()
{
  FEDRawDataCollection* result = 0;
  waitReadSem();
  //do not read data if destructor is called
  if (destroying_) return 0;
  lock();
  result = eventQueue_[readIndex_];
  eventQueue_[readIndex_]=0;
  readIndex_=(readIndex_+1)%queueSize_;
  unlock();
  
  postWriteSem();
  return result;
}


//______________________________________________________________________________
FEDRawDataCollection* PlaybackRawDataProvider::getFEDRawData(unsigned int& runNumber,
							     unsigned int& evtNumber)
{

  FEDRawDataCollection* result = 0;
  waitReadSem();
  //do not read data if destructor is called
  if (destroying_) return 0;
  lock();
  runNumber=runNumber_[readIndex_];
  evtNumber=evtNumber_[readIndex_];
  result=eventQueue_[readIndex_];
  //  assert(0!=result);
  eventQueue_[readIndex_]=0;
  readIndex_=(readIndex_+1)%queueSize_;

  unlock();
  
  postWriteSem();
  return result;
}


//______________________________________________________________________________
void PlaybackRawDataProvider::sem_print()
{
  lock();
  int wsem,rsem;
  sem_getvalue(&writeSem_,&wsem);
  sem_getvalue(&readSem_,&rsem);
  cout<<"sem_print():"
      <<" wsem="<<wsem
      <<" rsem="<<rsem
      <<" writeIndex="<<writeIndex_
      <<" readIndex="<<readIndex_
      <<endl;
  unlock();
}

void PlaybackRawDataProvider::setFreeToEof()
{ 
  //  cout << "  PlaybackRawDataProvider::setFreeToEof()" << endl;
  freeToEof_ = true; 
  //  cout << "  PlaybackRawDataProvider::setFreeToEof() call postReadSem" << endl;  
  postWriteSem(); 
}

bool PlaybackRawDataProvider::areFilesClosed()
{
  return filesClosed_;
}
////////////////////////////////////////////////////////////////////////////////
// framework module implementation macro
////////////////////////////////////////////////////////////////////////////////

