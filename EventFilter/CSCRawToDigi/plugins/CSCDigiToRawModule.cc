/** \file
 *  \author A. Tumanov - Rice
 */

#include "EventFilter/CSCRawToDigi/src/CSCDigiToRaw.h"
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
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
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
  unsigned int theFormatVersion;  // Select which version of data format to use Pre-LS1: 2005, Post-LS1: 2013
  bool usePreTriggers;            // Select if to use Pre-Triigers CLCT digis
  bool packEverything_;           // bypass all cuts and (pre)trigger requirements
  bool useGEMs_;

  std::unique_ptr<const CSCDigiToRaw> packer_;

  edm::EDGetTokenT<CSCWireDigiCollection> wd_token;
  edm::EDGetTokenT<CSCStripDigiCollection> sd_token;
  edm::EDGetTokenT<CSCComparatorDigiCollection> cd_token;
  edm::EDGetTokenT<CSCALCTDigiCollection> al_token;
  edm::EDGetTokenT<CSCCLCTDigiCollection> cl_token;
  edm::EDGetTokenT<CSCCLCTPreTriggerCollection> pr_token;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> co_token;
  edm::ESGetToken<CSCChamberMap, CSCChamberMapRcd> cham_token;
  edm::EDGetTokenT<GEMPadDigiClusterCollection> gem_token;

  edm::EDPutTokenT<FEDRawDataCollection> put_token_;
};

CSCDigiToRawModule::CSCDigiToRawModule(const edm::ParameterSet& pset) : packer_(std::make_unique<CSCDigiToRaw>(pset)) {
  //theStrip = pset.getUntrackedParameter<string>("DigiCreator", "cscunpacker");

  theFormatVersion = pset.getParameter<unsigned int>("useFormatVersion");  // pre-LS1 - '2005'. post-LS1 - '2013'
  usePreTriggers = pset.getParameter<bool>("usePreTriggers");              // disable checking CLCT PreTriggers digis
  packEverything_ = pset.getParameter<bool>("packEverything");  // don't check for consistency with trig primitives
                                                                // overrides usePreTriggers

  useGEMs_ = pset.getParameter<bool>("useGEMs");
  wd_token = consumes<CSCWireDigiCollection>(pset.getParameter<edm::InputTag>("wireDigiTag"));
  sd_token = consumes<CSCStripDigiCollection>(pset.getParameter<edm::InputTag>("stripDigiTag"));
  cd_token = consumes<CSCComparatorDigiCollection>(pset.getParameter<edm::InputTag>("comparatorDigiTag"));
  if (usePreTriggers) {
    pr_token = consumes<CSCCLCTPreTriggerCollection>(pset.getParameter<edm::InputTag>("preTriggerTag"));
  }
  al_token = consumes<CSCALCTDigiCollection>(pset.getParameter<edm::InputTag>("alctDigiTag"));
  cl_token = consumes<CSCCLCTDigiCollection>(pset.getParameter<edm::InputTag>("clctDigiTag"));
  co_token = consumes<CSCCorrelatedLCTDigiCollection>(pset.getParameter<edm::InputTag>("correlatedLCTDigiTag"));
  cham_token = esConsumes<CSCChamberMap, CSCChamberMapRcd>();
  if (useGEMs_) {
    gem_token = consumes<GEMPadDigiClusterCollection>(pset.getParameter<edm::InputTag>("padDigiClusterTag"));
  }
  put_token_ = produces<FEDRawDataCollection>("CSCRawData");
}

void CSCDigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  /*** From python/cscPacker_cfi.py
    wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi"),
    stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi"),
    comparatorDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCComparatorDigi"),
    alctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    clctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    preTriggerTag = cms.InputTag("simCscTriggerPrimitiveDigis"),
    correlatedLCTDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED"),
    # if min parameter = -999 always accept
    alctWindowMin = cms.int32(-3),
    alctWindowMax = cms.int32(3),
    clctWindowMin = cms.int32(-3),
    clctWindowMax = cms.int32(3),
    preTriggerWindowMin = cms.int32(-3),
    preTriggerWindowMax = cms.int32(1)
*/

  edm::ParameterSetDescription desc;

  desc.add<unsigned int>("useFormatVersion", 2005)
      ->setComment("Set to 2005 for pre-LS1 CSC data format, 2013 - new post-LS1 CSC data format");
  desc.add<bool>("usePreTriggers", true)->setComment("Set to false if CSCCLCTPreTrigger digis are not available");
  desc.add<bool>("packEverything", false)
      ->setComment("Set to true to disable trigger-related constraints on readout data");
  desc.add<bool>("useGEMs", false)->setComment("Pack GEM trigger data");

  desc.add<edm::InputTag>("wireDigiTag", edm::InputTag("simMuonCSCDigis", "MuonCSCWireDigi"));
  desc.add<edm::InputTag>("stripDigiTag", edm::InputTag("simMuonCSCDigis", "MuonCSCStripDigi"));
  desc.add<edm::InputTag>("comparatorDigiTag", edm::InputTag("simMuonCSCDigis", "MuonCSCComparatorDigi"));
  desc.add<edm::InputTag>("alctDigiTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("clctDigiTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("preTriggerTag", edm::InputTag("simCscTriggerPrimitiveDigis"));
  desc.add<edm::InputTag>("correlatedLCTDigiTag", edm::InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED"));
  desc.add<edm::InputTag>("padDigiClusterTag", edm::InputTag("simMuonGEMPadDigiClusters"));

  desc.add<int32_t>("alctWindowMin", -3)->setComment("If min parameter = -999 always accept");
  desc.add<int32_t>("alctWindowMax", 3);
  desc.add<int32_t>("clctWindowMin", -3)->setComment("If min parameter = -999 always accept");
  desc.add<int32_t>("clctWindowMax", 3);
  desc.add<int32_t>("preTriggerWindowMin", -3)->setComment("If min parameter = -999 always accept");
  desc.add<int32_t>("preTriggerWindowMax", 1);

  descriptions.add("cscPacker", desc);
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
  edm::Handle<CSCCLCTPreTriggerCollection> preTriggers;
  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedLCTDigis;
  edm::Handle<GEMPadDigiClusterCollection> padDigiClusters;

  e.getByToken(wd_token, wireDigis);
  e.getByToken(sd_token, stripDigis);
  e.getByToken(cd_token, comparatorDigis);
  e.getByToken(al_token, alctDigis);
  e.getByToken(cl_token, clctDigis);
  if (usePreTriggers)
    e.getByToken(pr_token, preTriggers);
  e.getByToken(co_token, correlatedLCTDigis);
  if (useGEMs_) {
    e.getByToken(gem_token, padDigiClusters);
  }
  // Create the packed data
  packer_->createFedBuffers(*stripDigis,
                            *wireDigis,
                            *comparatorDigis,
                            *alctDigis,
                            *clctDigis,
                            *preTriggers,
                            *correlatedLCTDigis,
                            *padDigiClusters,
                            fed_buffers,
                            theMapping,
                            e,
                            theFormatVersion,
                            usePreTriggers,
                            useGEMs_,
                            packEverything_);

  // put the raw data to the event
  e.emplace(put_token_, std::move(fed_buffers));
}

DEFINE_FWK_MODULE(CSCDigiToRawModule);
