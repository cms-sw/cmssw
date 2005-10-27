#ifndef SiStripDigiToRawModule_H
#define SiStripDigiToRawModule_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include <memory>
#include <string>

class SiStripDigiToRaw;
class SiStripUtility;

/**
   \class SiStripDigiToRawModule 
   \brief A plug-in module that performs DigiToRaw conversion.
   \author R.Bainbridge
   \version 0.1
   \date 05/09/05

   A plug-in module that performs DigiToRaw conversion. 
   Input (from Event): StripDigiCollection. 
   Output (EDProduct): FEDRawDataCollection. 
   Nota bene: this is a PROTOTYPE IMPLEMENTATION!
*/
class SiStripDigiToRawModule : public edm::EDProducer {
  
public:
  
  /** Constructor. */
  explicit SiStripDigiToRawModule( const edm::ParameterSet& );
  /** Destructor. */
  ~SiStripDigiToRawModule();

  /** Some initialisation. Retrieves cabling map from
      EventSetup. Creates DigiToRaw converter object. */
  virtual void beginJob( const edm::EventSetup& );
  /** Currently does nothing. */
  virtual void endJob();
  
  /** Retrieves a StripDigiCollection from the Event, creates a
      FEDRawDataCollection (EDProduct) using the DigiToRaw converter,
      and attaches it to the Event. */
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
private:
  
  /** DigiToRaw converter that creates FEDRawData objects containing
      FED buffers, based on the input of StripDigis. */
  SiStripDigiToRaw* digiToRaw_;
  /** Utility class providing dummy digis, FED buffers, cabling map. */
  SiStripUtility* utility_;
  int numDigis;

  /** Event counter. */
  unsigned long event_;

  /** Defines the FED readout mode (ZS, VR, PR or SM). */
  std::string fedReadoutMode_;
  /** Defines the FED readout path (VME or SLINK). */
  std::string fedReadoutPath_;
  
  /** Defines verbosity level for this class (0=silent -> 3=debug). */
  int verbosity_;

};

#endif // SiStripDigiToRawModule_H

