#ifndef EventFilter_SiStripRawToDigiModule_H
#define EventFilter_SiStripRawToDigiModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include <string>

class SiStripRawToDigi;

/**
   \class SiStripRawToDigiModule 
   \brief A plug-in module that takes a FEDRawDataCollection as input
   from the Event and creates an EDProduct in the form of a
   StripDigiCollection.
   \author R.Bainbridge
*/
class SiStripRawToDigiModule : public edm::EDProducer {
  
 public:

  /** Constructor creates RawToDigi unpacking object. */
  SiStripRawToDigiModule( const edm::ParameterSet& );
  ~SiStripRawToDigiModule();

  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  /** Retrieves a FEDRawDataCollection from the Event, creates an
      EDProduct in the form of a StripDigiCollection (using the
      SiStripRawToDigi class) and attaches the digi collection to the
      Event. */
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 
  
  /** RawToDigi converter class that creates digis. */
  SiStripRawToDigi* rawToDigi_;
  /** Event counter. */
  unsigned long eventCounter_;
  /** Verbosity level for this class (0=silent, 3=debug). */
  int verbosity_;

};

#endif // EventFilter_SiStripRawToDigiModule_H

