#ifndef EventFilter_SiStripFEDRawDataToEvent_H
#define EventFilter_SiStripFEDRawDataToEvent_H

#include "FWCore/Framework/interface/EDProducer.h"

 /**
    \class SiStripFEDRawDataToEvent 
    \brief Creates dummy FED buffers and wraps using FEDRawData before
    attaching a collection to the Event. Allows to test the final
    RawToDigi unpacker.
    \author R.Bainbridge, M.Wingham
*/
class SiStripFEDRawDataToEvent : public edm::EDProducer {
  
 public:
  
  SiStripFEDRawDataToEvent( const edm::ParameterSet& );
  ~SiStripFEDRawDataToEvent();
  
  virtual void beginJob( const edm::EventSetup& ) {;}
  virtual void endJob() {;}
  
  /** Creates "dummy" FED buffers and attaches a FEDRawDataCollection
      to the Event. */
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private: 
  
  /** Event counter. */
  unsigned long eventCounter_;
  /** Defines the FED readout mode (ZS, VR, PR or SM). */
  std::string fedReadoutMode_;
  /** Verbosity level for this class (0=silent, 3=debug). */
  int verbosity_;

};

#endif // EventFilter_SiStripFEDRawDataToEvent_H
