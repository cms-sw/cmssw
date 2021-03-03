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

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

class SiStripRegFEDSelector : public edm::EDProducer {
public:
  SiStripRegFEDSelector(const edm::ParameterSet&);
  ~SiStripRegFEDSelector() override;

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_seed_;
  const edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  const edm::ESGetToken<SiStripRegionCabling, SiStripRegionCablingRcd> tok_strip_;
  const double delta_;
};

#endif
