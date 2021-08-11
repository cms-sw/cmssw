/** \file
 *  \author A. Tumanov - Rice
 */

#include "EventFilter/CSCRawToDigi/interface/CSCDigiToRaw.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/DataRecord/interface/CSCChamberMapRcd.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

namespace edm {
  class ConfigurationDescriptions;
}

class CSCDigiToRaw;

class CSCDigiToRawModule : public edm::global::EDProducer<> {
public:
  /// Constructor
  CSCDigiToRawModule(const edm::ParameterSet& pset);

  // Operations
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // Fill parameters descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool usePreTriggers_;  // Select if to use Pre-Triggers CLCT digis
  bool useGEMs_;
  bool useCSCShowers_;

  std::unique_ptr<const CSCDigiToRaw> packer_;

  edm::EDGetTokenT<CSCWireDigiCollection> wd_token;
  edm::EDGetTokenT<CSCStripDigiCollection> sd_token;
  edm::EDGetTokenT<CSCComparatorDigiCollection> cd_token;
  edm::EDGetTokenT<CSCALCTDigiCollection> al_token;
  edm::EDGetTokenT<CSCCLCTDigiCollection> cl_token;
  edm::EDGetTokenT<CSCCLCTPreTriggerCollection> pr_token;
  edm::EDGetTokenT<CSCCLCTPreTriggerDigiCollection> prdigi_token;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> co_token;
  edm::EDGetTokenT<CSCShowerDigiCollection> shower_token;
  edm::ESGetToken<CSCChamberMap, CSCChamberMapRcd> cham_token;
  edm::EDGetTokenT<GEMPadDigiClusterCollection> gem_token;

  edm::EDPutTokenT<FEDRawDataCollection> put_token_;
};

CSCDigiToRawModule::CSCDigiToRawModule(const edm::ParameterSet& pset) : packer_(std::make_unique<CSCDigiToRaw>(pset)) {
  usePreTriggers_ = pset.getParameter<bool>("usePreTriggers");  // disable checking CLCT PreTriggers digis

  useGEMs_ = pset.getParameter<bool>("useGEMs");
  useCSCShowers_ = pset.getParameter<bool>("useCSCShowers");
  wd_token = consumes<CSCWireDigiCollection>(pset.getParameter<edm::InputTag>("wireDigiTag"));
  sd_token = consumes<CSCStripDigiCollection>(pset.getParameter<edm::InputTag>("stripDigiTag"));
  cd_token = consumes<CSCComparatorDigiCollection>(pset.getParameter<edm::InputTag>("comparatorDigiTag"));
  if (usePreTriggers_) {
    pr_token = consumes<CSCCLCTPreTriggerCollection>(pset.getParameter<edm::InputTag>("preTriggerTag"));
    prdigi_token = consumes<CSCCLCTPreTriggerDigiCollection>(pset.getParameter<edm::InputTag>("preTriggerDigiTag"));
  }
  al_token = consumes<CSCALCTDigiCollection>(pset.getParameter<edm::InputTag>("alctDigiTag"));
  cl_token = consumes<CSCCLCTDigiCollection>(pset.getParameter<edm::InputTag>("clctDigiTag"));
  co_token = consumes<CSCCorrelatedLCTDigiCollection>(pset.getParameter<edm::InputTag>("correlatedLCTDigiTag"));
  cham_token = esConsumes<CSCChamberMap, CSCChamberMapRcd>();
  if (useGEMs_) {
    gem_token = consumes<GEMPadDigiClusterCollection>(pset.getParameter<edm::InputTag>("padDigiClusterTag"));
  }
  if (useCSCShowers_) {
    shower_token = consumes<CSCShowerDigiCollection>(pset.getParameter<edm::InputTag>("showerDigiTag"));
  }
  put_token_ = produces<FEDRawDataCollection>("CSCRawData");
}

void CSCDigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<unsigned int>("formatVersion", 2005)
      ->setComment("Set to 2005 for pre-LS1 CSC data format, 2013 - new post-LS1 CSC data format");
  desc.add<bool>("usePreTriggers", true)->setComment("Set to false if CSCCLCTPreTrigger digis are not available");
  desc.add<bool>("packEverything", false)
      ->setComment("Set to true to disable trigger-related constraints on readout data");
  desc.add<bool>("useGEMs", false)->setComment("Pack GEM trigger data");
  desc.add<bool>("useCSCShowers", false)->setComment("Pack CSC shower trigger data");

  desc.add<edm::InputTag>("wireDigiTag", edm::InputTag("simMuonCSCDigis", "MuonCSCWireDigi"));
  desc.add<edm::InputTag>("stripDigiTag", edm::InputTag("simMuonCSCDigis", "MuonCSCStripDigi"));
  desc.add<edm::InputTag>("comparatorDigiTag", edm::InputTag("simMuonCSCDigis", "MuonCSCComparatorDigi"));
  desc.add<edm::InputTag>("alctDigiTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("clctDigiTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("preTriggerTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("preTriggerDigiTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("correlatedLCTDigiTag", edm::InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED"));
  desc.add<edm::InputTag>("padDigiClusterTag", edm::InputTag("simMuonGEMPadDigiClusters"));
  desc.add<edm::InputTag>("showerDigiTag", edm::InputTag("simCscTriggerPrimitiveDigis"));

  desc.add<int32_t>("alctWindowMin", -3)->setComment("If min parameter = -999 always accept");
  desc.add<int32_t>("alctWindowMax", 3);
  desc.add<int32_t>("clctWindowMin", -3)->setComment("If min parameter = -999 always accept");
  desc.add<int32_t>("clctWindowMax", 3);
  desc.add<int32_t>("preTriggerWindowMin", -3)->setComment("If min parameter = -999 always accept");
  desc.add<int32_t>("preTriggerWindowMax", 1);

  descriptions.add("cscPackerDef", desc);
}

void CSCDigiToRawModule::produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const {
  ///reverse mapping for packer
  edm::ESHandle<CSCChamberMap> hcham = c.getHandle(cham_token);
  const CSCChamberMap* theMapping = hcham.product();

  FEDRawDataCollection fed_buffers;

  // Take digis from the event
  edm::Handle<CSCWireDigiCollection> wireDigis;
  edm::Handle<CSCStripDigiCollection> stripDigis;
  edm::Handle<CSCComparatorDigiCollection> comparatorDigis;
  edm::Handle<CSCALCTDigiCollection> alctDigis;
  edm::Handle<CSCCLCTDigiCollection> clctDigis;
  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedLCTDigis;

  // collections that are always packed
  e.getByToken(wd_token, wireDigis);
  e.getByToken(sd_token, stripDigis);
  e.getByToken(cd_token, comparatorDigis);
  e.getByToken(al_token, alctDigis);
  e.getByToken(cl_token, clctDigis);
  e.getByToken(co_token, correlatedLCTDigis);

  // packing with pre-triggers
  CSCCLCTPreTriggerCollection const* preTriggersPtr = nullptr;
  CSCCLCTPreTriggerDigiCollection const* preTriggerDigisPtr = nullptr;
  if (usePreTriggers_) {
    preTriggersPtr = &e.get(pr_token);
    preTriggerDigisPtr = &e.get(prdigi_token);
  }

  // collections that are packed optionally

  // packing of GEM hits
  const GEMPadDigiClusterCollection* padDigiClustersPtr = nullptr;
  if (useGEMs_) {
    padDigiClustersPtr = &e.get(gem_token);
  }

  // packing of CSC shower digis
  const CSCShowerDigiCollection* cscShowerDigisPtr = nullptr;
  if (useCSCShowers_) {
    cscShowerDigisPtr = &e.get(shower_token);
  }

  // Create the packed data
  packer_->createFedBuffers(*stripDigis,
                            *wireDigis,
                            *comparatorDigis,
                            *alctDigis,
                            *clctDigis,
                            preTriggersPtr,
                            preTriggerDigisPtr,
                            *correlatedLCTDigis,
                            cscShowerDigisPtr,
                            padDigiClustersPtr,
                            fed_buffers,
                            theMapping,
                            e.id());

  // put the raw data to the event
  e.emplace(put_token_, std::move(fed_buffers));
}

DEFINE_FWK_MODULE(CSCDigiToRawModule);
