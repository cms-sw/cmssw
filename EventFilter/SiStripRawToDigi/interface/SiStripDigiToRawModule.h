#ifndef SiStripDigiToRawModule_H
#define SiStripDigiToRawModule_H

// system include files
#include <memory>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
#include <string>

class SiStripDigiToRaw;
class SiStripUtility;

/**
   \class SiStripDigiToRawModule 
   \brief Short description here
   \author R.Bainbridge
   \version 0.1
   \date 09/08/05
   
   Long description here.
*/
class SiStripDigiToRawModule : public edm::EDProducer {
  
public:
  
  /** */
  explicit SiStripDigiToRawModule( const edm::ParameterSet& );
  /** */
  ~SiStripDigiToRawModule();

  /** */
  virtual void beginJob( const edm::EventSetup& );
  /** */
  virtual void endJob();
  
  /** */
  virtual void produce( edm::Event&, const edm::EventSetup& );
  
private:
  
  /** converter from Digi's to FED buffers */
  SiStripDigiToRaw* digiToRaw_;
  /** utility class providing dummy Digis, FED buffers and cabling map */
  SiStripUtility* utility_;

  /** event counter */
  unsigned long event_;

  /** defines whether the FED readout path is via VME or SLINK */
  std::string fedReadoutPath_;
  /** defines level of verbosity for this class (0=silent -> 3=debug) */
  unsigned short verbosity_;
  
};

#endif // SiStripDigiToRawModule_H

