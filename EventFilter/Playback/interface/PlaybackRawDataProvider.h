#ifndef PLAYBACKRAWDATAPROVIDER_H
#define PLAYBACKRAWDATAPROVIDER_H 1

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <semaphore.h>


class PlaybackRawDataProvider : public edm::EDAnalyzer
{
public:
  //
  // construction/destruction
  //
  explicit PlaybackRawDataProvider(const edm::ParameterSet&);
  ~PlaybackRawDataProvider();
  
  
  //
  // member functions
  //
  
  // EDAnalyzer interface
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // provide cached fed collection (and run/evt number, if needed!)
  FEDRawDataCollection* getFEDRawData();
  FEDRawDataCollection* getFEDRawData(unsigned int& runNumber,
				      unsigned int& evtNumber);
  
  
  static PlaybackRawDataProvider* instance();
  
  
private:
  //
  // member data
  //
  static PlaybackRawDataProvider* instance_;
  
  FEDRawDataCollection*           rawData_;
  unsigned int                    runNumber_;
  unsigned int                    evtNumber_;
  unsigned int                    count_;
  
  sem_t                           mutex1_;
  sem_t                           mutex2_;
  
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
