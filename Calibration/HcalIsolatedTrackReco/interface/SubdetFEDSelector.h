#ifndef Calibration_SubdetFEDSelector_h
#define Calibration_SubdetFEDSelector_h

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/RawDataCollector/interface/RawDataFEDSelector.h"

class SubdetFEDSelector : public edm::EDProducer {
public:
  SubdetFEDSelector(const edm::ParameterSet&);
  ~SubdetFEDSelector() override;

  bool getEcal_;
  bool getHcal_;
  bool getStrip_;
  bool getPixel_;
  bool getMuon_;
  bool getTrigger_;

  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;

private:
  void beginJob() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
};

#endif
