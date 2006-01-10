#ifndef EventFilter_SiStripDigiToRawModule_H
#define EventFilter_SiStripDigiToRawModule_H

#include "FWCore/Framework/interface/EDProducer.h"
#include <string>

class SiStripDigiToRaw;
class SiStripUtility;

/**
   \class SiStripDigiToRawModule 
   \brief A plug-in module that takes a StripDigiCollection as input
   from the Event and creates an EDProduct in the form of a
   FEDRawDataCollection.
   \author R.Bainbridge
*/
class SiStripDigiToRawModule : public edm::EDProducer {
  
 public:
  
  /** Constructor creates DigiToRaw formatting object. */
  SiStripDigiToRawModule( const edm::ParameterSet& );
  ~SiStripDigiToRawModule();
  
  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  /** Retrieves a StripDigiCollection from the Event, creates an
      EDProduct in the form of a FEDRawDataCollection (using the
      SiStripDigiToRaw class) and attaches the collection to the
      Event. */
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private:
  
  /** RawToDigi class that creates FED buffers using digis. */
  SiStripDigiToRaw* digiToRaw_;
  /** Utility class providing digis. */
  SiStripUtility* utility_;
  /** Event counter. */
  unsigned long eventCounter_;
  /** Label used to identify EDProduct within Event. */
  std::string productLabel_;
  /** Verbosity level for this class (0=silent, 3=debug). */
  int verbosity_;

};

#endif // EventFilter_SiStripDigiToRawModule_H

