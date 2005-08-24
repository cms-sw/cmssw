#ifndef SiStripRawToDigiModule_H
#define SiStripRawToDigiModule_H

// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include <string>

class SiStripRawToDigi;
class SiStripUtility;

/**
   \class SiStripRawToDigiModule 
   \brief A plug-in EDProducer module that takes a
   FEDRawDataCollection as input and creates an EDProduct in the form
   of a StripDigiCollection. 
   \author R.Bainbridge
   \version 0.1
   \date 09/08/05
   
   A plug-in EDProducer module that takes a FEDRawDataCollection as
   input and creates an EDProduct in the form of a
   StripDigiCollection. 
   Nota bene: this is a PROTOTYPE IMPLEMENTATION!
*/
class SiStripRawToDigiModule : public edm::EDProducer {
  
public:
  
  /** */
  explicit SiStripRawToDigiModule( const edm::ParameterSet& );
  /** */
  ~SiStripRawToDigiModule();

  /** */
  virtual void beginJob( const edm::EventSetup& );
  /** */
  virtual void endJob();
  
  /** method that retrieves a FEDRawDataCollection from the Event and
      creates an EDProduct in the form of a StripDigiCollection. */
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
private:
  
  /** object that converts FEDRawData objects to StripDigi's */
  SiStripRawToDigi* rawToDigi_;
  /** utility class providing dummy Digis, FED buffers, cabling map */
  SiStripUtility* utility_;

  /** event counter */
  unsigned long event_;

  /** defines the FED readout mode: ZS, VR, PR or SM */
  std::string fedReadoutMode_;
  /** defines the FED readout path: VME or SLINK */
  std::string fedReadoutPath_;
  /** defines level of verbosity for this class (0=silent -> 3=debug) */
  int verbosity_;

  long ndigis_;
  
};

#endif // SiStripRawToDigiModule_H

