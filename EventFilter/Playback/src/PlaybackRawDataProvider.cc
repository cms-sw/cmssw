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

#include <iostream>


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
{
  queueSize_=iConfig.getUntrackedParameter<unsigned int>("QueueSize",128);
  sem_init(&lock_,0,1);
  sem_init(&writeSem_,0,queueSize_);
  sem_init(&readSem_,0,0);
  runNumber_ =new unsigned int[queueSize_];
  evtNumber_ =new unsigned int[queueSize_];
  eventQueue_=new FEDRawDataCollection*[queueSize_];
  for (unsigned int i=0;i<queueSize_;i++) eventQueue_[i]=0;
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
  instance_=0;
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void PlaybackRawDataProvider::analyze(const Event& iEvent,
				      const EventSetup& iSetup)
{
  waitWriteSem();
  
  runNumber_[writeIndex_]=iEvent.id().run();
  evtNumber_[writeIndex_]=iEvent.id().event();

  Handle<FEDRawDataCollection> pRawData;
  iEvent.getByType(pRawData);
  
  if (!pRawData.isValid()) {
    edm::LogError("InvalidHandle")<<"no raw data found!"<<endl;
    return;
  }
  
  
  // copy the raw data collection into rawData_, retrievable via getFEDRawData()
  assert(0==eventQueue_[writeIndex_]);
  eventQueue_[writeIndex_]=new FEDRawDataCollection();
  for (unsigned int i=0;i<(unsigned int)FEDNumbering::lastFEDId()+1;i++) {
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
void PlaybackRawDataProvider::beginJob(const EventSetup&)
{
  
}


//______________________________________________________________________________
void PlaybackRawDataProvider::endJob()
{
  edm::LogInfo("Summary")<<count_<<" events read."<<endl;
}


//______________________________________________________________________________
FEDRawDataCollection* PlaybackRawDataProvider::getFEDRawData()
{
  waitReadSem();

  lock();
  FEDRawDataCollection* result=eventQueue_[readIndex_];
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
  waitReadSem();

  lock();
  
  runNumber=runNumber_[readIndex_];
  evtNumber=evtNumber_[readIndex_];
  FEDRawDataCollection* result=eventQueue_[readIndex_];
  assert(0!=result);
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


////////////////////////////////////////////////////////////////////////////////
// framework module implementation macro
////////////////////////////////////////////////////////////////////////////////

DEFINE_FWK_MODULE(PlaybackRawDataProvider);
