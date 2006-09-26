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
  : rawData_(0)
  , runNumber_(0)
  , evtNumber_(0)
  , count_(0)
{
  sem_init(&mutex1_,0,1);
  sem_init(&mutex2_,0,0);
  instance_=this;
}


//______________________________________________________________________________
PlaybackRawDataProvider::~PlaybackRawDataProvider()
{
  instance_=0;
}


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void PlaybackRawDataProvider::analyze(const Event& iEvent,
				    const EventSetup& iSetup)
{
  sem_wait(&mutex1_);

  runNumber_=iEvent.id().run();
  evtNumber_=iEvent.id().event();

  Handle<FEDRawDataCollection> pRawData;
  iEvent.getByType(pRawData);
  
  if (!pRawData.isValid()) {
    edm::LogError("***")<<"no raw data found!"<<endl;
    return;
  }
  
  if (0!=rawData_) delete rawData_;
  
  // copy the raw data collection into rawData_, retrievable via getFEDRawData()
  rawData_=new FEDRawDataCollection();
  for (unsigned int i=0;i<(unsigned int)FEDNumbering::lastFEDId()+1;i++) {
    unsigned int fedSize=pRawData->FEDData(i).size();
    if (fedSize>0) {
      FEDRawData& fedData=rawData_->FEDData(i);
      fedData.resize(fedSize);
      memcpy(fedData.data(),pRawData->FEDData(i).data(),fedSize);
    }
  }

  count_++;

  sem_post(&mutex2_);
  
  return;
}


//______________________________________________________________________________
void PlaybackRawDataProvider::beginJob(const EventSetup&)
{
  
}


//______________________________________________________________________________
void PlaybackRawDataProvider::endJob()
{
  
}


//______________________________________________________________________________
FEDRawDataCollection* PlaybackRawDataProvider::getFEDRawData()
{
  sem_wait(&mutex2_);
  
  FEDRawDataCollection* result=rawData_;
  rawData_=0;
  
  sem_post(&mutex1_);
  
  return result;
}


//______________________________________________________________________________
FEDRawDataCollection* PlaybackRawDataProvider::getFEDRawData(unsigned int& runNumber,
							   unsigned int& evtNumber)
{
  sem_wait(&mutex2_);
  
  runNumber=runNumber_;
  evtNumber=evtNumber_;
  FEDRawDataCollection* result=rawData_;
  rawData_=0;
  
  sem_post(&mutex1_);
  
  return result;
}


////////////////////////////////////////////////////////////////////////////////
// framework module implementation macro
////////////////////////////////////////////////////////////////////////////////

DEFINE_FWK_MODULE(PlaybackRawDataProvider)
