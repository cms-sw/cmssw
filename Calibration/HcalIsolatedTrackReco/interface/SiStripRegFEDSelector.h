#ifndef Calibration_SiStripRegFEDSelector_h
#define Calibration_SiStripRegFEDSelector_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"

class SiStripRegFEDSelector : public edm::EDProducer {
public:
  SiStripRegFEDSelector(const edm::ParameterSet&);
  ~SiStripRegFEDSelector();

  edm::ESHandle<SiStripRegionCabling> strip_cabling;

  double delta_;
  
  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_seed_;

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

};

#endif
