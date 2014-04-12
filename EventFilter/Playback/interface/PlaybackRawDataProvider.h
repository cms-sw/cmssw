#ifndef PLAYBACKRAWDATAPROVIDER_H
#define PLAYBACKRAWDATAPROVIDER_H 1

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <semaphore.h>

class FEDRawDataCollection;

class PlaybackRawDataProvider : public edm::EDAnalyzer
{
public:
  //
  // construction/destruction
  //
  explicit PlaybackRawDataProvider(const edm::ParameterSet&);
  virtual ~PlaybackRawDataProvider();
  
  // EDAnalyzer interface
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual void respondToCloseInputFile(edm::FileBlock const& fb);

  // provide cached fed collection (and run/evt number, if needed!)
  virtual FEDRawDataCollection* getFEDRawData();
  virtual FEDRawDataCollection* getFEDRawData(unsigned int& runNumber,
					      unsigned int& evtNumber);
  void setFreeToEof();
  bool areFilesClosed();

  static PlaybackRawDataProvider* instance();
  
  
private:

  static PlaybackRawDataProvider* instance_;
    //
  // private member functions
  //
  void lock()         { sem_wait(&lock_); }
  void unlock()       { sem_post(&lock_); }
  void waitWriteSem() { sem_wait(&writeSem_); }
  void postWriteSem() { sem_post(&writeSem_); }
  void waitReadSem()  { sem_wait(&readSem_);}
  void postReadSem()  { sem_post(&readSem_); }


  void sem_print();
  
  
private:
  //
  // member data
  //
  unsigned int           queueSize_;
  FEDRawDataCollection **eventQueue_;
  unsigned int          *runNumber_;
  unsigned int          *evtNumber_;
  unsigned int           count_;
  
  sem_t                  lock_;
  sem_t                  writeSem_;
  sem_t                  readSem_;
  unsigned int           writeIndex_;
  unsigned int           readIndex_;
  bool                   freeToEof_;
  bool                   filesClosed_;
  bool                   destroying_;
};


//
// implementation of inline functions
//

//______________________________________________________________________________
inline
PlaybackRawDataProvider* PlaybackRawDataProvider::instance()
{
  return instance_;
}


#endif
