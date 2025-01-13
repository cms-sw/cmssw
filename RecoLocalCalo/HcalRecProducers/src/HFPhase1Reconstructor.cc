// -*- C++ -*-
//
// Package:    RecoLocalCalo/HcalRecProducers
// Class:      HFPhase1Reconstructor
//
/**\class HFPhase1Reconstructor HFPhase1Reconstructor.cc RecoLocalCalo/HcalRecProducers/src/HFPhase1Reconstructor.cc

 Description: Phase 1 HF reco with QIE 10 and split-anode readout

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Igor Volobouev
//         Created:  Thu, 25 May 2016 00:17:51 GMT
//
//

// system include files

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HFPhase1PMTParams.h"
#include "CondFormats/DataRecord/interface/HFPhase1PMTParamsRcd.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

// Noise cleanup algos
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHF_PETalgorithm.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalHF_S9S1algorithm.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HFStripFilter.h"

// Parser for Phase 1 HF reco algorithms
#include "RecoLocalCalo/HcalRecAlgos/interface/parseHFPhase1AlgoDescription.h"

//
// class declaration
//
class HFPhase1Reconstructor : public edm::stream::EDProducer<> {
public:
  explicit HFPhase1Reconstructor(const edm::ParameterSet&);
  ~HFPhase1Reconstructor() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  // Configuration parameters
  std::string algoConfigClass_;
  bool setNoiseFlags_;
  bool runHFStripFilter_;
  bool useChannelQualityFromDB_;
  bool checkChannelQualityForDepth3and4_;

  // Other members
  edm::EDGetTokenT<HFPreRecHitCollection> tok_PreRecHit_;
  std::unique_ptr<AbsHFPhase1Algo> reco_;
  std::unique_ptr<AbsHcalAlgoData> recoConfig_;
  edm::ESGetToken<HFPhase1PMTParams, HFPhase1PMTParamsRcd> recoConfigToken_;

  // Noise cleanup algos
  std::unique_ptr<HcalHF_S9S1algorithm> hfS9S1_;
  std::unique_ptr<HcalHF_S9S1algorithm> hfS8S1_;
  std::unique_ptr<HcalHF_PETalgorithm> hfPET_;
  std::unique_ptr<HFStripFilter> hfStripFilter_;

  // ES tokens
  edm::ESGetToken<HcalDbService, HcalDbRecord> conditionsToken_;
  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> qualToken_;
  edm::ESGetToken<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd> sevToken_;
};

//
// constructors and destructor
//
HFPhase1Reconstructor::HFPhase1Reconstructor(const edm::ParameterSet& conf)
    : algoConfigClass_(conf.getParameter<std::string>("algoConfigClass")),
      setNoiseFlags_(conf.getParameter<bool>("setNoiseFlags")),
      runHFStripFilter_(conf.getParameter<bool>("runHFStripFilter")),
      useChannelQualityFromDB_(conf.getParameter<bool>("useChannelQualityFromDB")),
      checkChannelQualityForDepth3and4_(conf.getParameter<bool>("checkChannelQualityForDepth3and4")),
      reco_(parseHFPhase1AlgoDescription(conf.getParameter<edm::ParameterSet>("algorithm"))) {
  // Check that the reco algorithm has been successfully configured
  if (!reco_.get())
    throw cms::Exception("HFPhase1BadConfig") << "Invalid HFPhase1Reconstructor algorithm configuration" << std::endl;

  if (reco_->isConfigurable()) {
    if ("HFPhase1PMTParams" != algoConfigClass_) {
      throw cms::Exception("HBHEPhase1BadConfig")
          << "Invalid HBHEPhase1Reconstructor \"algoConfigClass\" parameter value \"" << algoConfigClass_ << '"'
          << std::endl;
    }
    recoConfigToken_ = esConsumes<edm::Transition::BeginRun>();
  }

  // Configure the noise cleanup algorithms
  if (setNoiseFlags_) {
    const edm::ParameterSet& psS9S1 = conf.getParameter<edm::ParameterSet>("S9S1stat");
    hfS9S1_ = std::make_unique<HcalHF_S9S1algorithm>(psS9S1.getParameter<std::vector<double>>("short_optimumSlope"),
                                                     psS9S1.getParameter<std::vector<double>>("shortEnergyParams"),
                                                     psS9S1.getParameter<std::vector<double>>("shortETParams"),
                                                     psS9S1.getParameter<std::vector<double>>("long_optimumSlope"),
                                                     psS9S1.getParameter<std::vector<double>>("longEnergyParams"),
                                                     psS9S1.getParameter<std::vector<double>>("longETParams"),
                                                     psS9S1.getParameter<int>("HcalAcceptSeverityLevel"),
                                                     psS9S1.getParameter<bool>("isS8S1"));

    const edm::ParameterSet& psS8S1 = conf.getParameter<edm::ParameterSet>("S8S1stat");
    hfS8S1_ = std::make_unique<HcalHF_S9S1algorithm>(psS8S1.getParameter<std::vector<double>>("short_optimumSlope"),
                                                     psS8S1.getParameter<std::vector<double>>("shortEnergyParams"),
                                                     psS8S1.getParameter<std::vector<double>>("shortETParams"),
                                                     psS8S1.getParameter<std::vector<double>>("long_optimumSlope"),
                                                     psS8S1.getParameter<std::vector<double>>("longEnergyParams"),
                                                     psS8S1.getParameter<std::vector<double>>("longETParams"),
                                                     psS8S1.getParameter<int>("HcalAcceptSeverityLevel"),
                                                     psS8S1.getParameter<bool>("isS8S1"));

    const edm::ParameterSet& psPET = conf.getParameter<edm::ParameterSet>("PETstat");
    hfPET_ = std::make_unique<HcalHF_PETalgorithm>(psPET.getParameter<std::vector<double>>("short_R"),
                                                   psPET.getParameter<std::vector<double>>("shortEnergyParams"),
                                                   psPET.getParameter<std::vector<double>>("shortETParams"),
                                                   psPET.getParameter<std::vector<double>>("long_R"),
                                                   psPET.getParameter<std::vector<double>>("longEnergyParams"),
                                                   psPET.getParameter<std::vector<double>>("longETParams"),
                                                   psPET.getParameter<int>("HcalAcceptSeverityLevel"),
                                                   psPET.getParameter<std::vector<double>>("short_R_29"),
                                                   psPET.getParameter<std::vector<double>>("long_R_29"));

    // Configure HFStripFilter
    if (runHFStripFilter_) {
      const edm::ParameterSet& psStripFilter = conf.getParameter<edm::ParameterSet>("HFStripFilter");
      hfStripFilter_ = HFStripFilter::parseParameterSet(psStripFilter);
    }
  }

  // Describe consumed data
  tok_PreRecHit_ = consumes<HFPreRecHitCollection>(conf.getParameter<edm::InputTag>("inputLabel"));

  // Register the product
  produces<HFRecHitCollection>();

  // ES tokens
  conditionsToken_ = esConsumes<HcalDbService, HcalDbRecord>();
  qualToken_ = esConsumes<HcalChannelQuality, HcalChannelQualityRcd>(edm::ESInputTag("", "withTopo"));
  sevToken_ = esConsumes<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd>();
}

void HFPhase1Reconstructor::beginRun(const edm::Run& r, const edm::EventSetup& es) {
  if (reco_->isConfigurable()) {
    recoConfig_ = std::make_unique<HFPhase1PMTParams>(es.getData(recoConfigToken_));
    if (!reco_->configure(recoConfig_.get()))
      throw cms::Exception("HFPhase1BadConfig")
          << "Failed to configure HFPhase1Reconstructor algorithm from EventSetup" << std::endl;
  }
}

// ------------ method called to produce the data  ------------
void HFPhase1Reconstructor::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  using namespace edm;

  // Fetch the calibrations
  const HcalDbService* conditions = &eventSetup.getData(conditionsToken_);
  const HcalChannelQuality* myqual = &eventSetup.getData(qualToken_);
  const HcalSeverityLevelComputer* mySeverity = &eventSetup.getData(sevToken_);

  // Get the input data
  Handle<HFPreRecHitCollection> preRecHits;
  e.getByToken(tok_PreRecHit_, preRecHits);

  // Create a new output collection
  std::unique_ptr<HFRecHitCollection> rec(std::make_unique<HFRecHitCollection>());
  rec->reserve(preRecHits->size());

  // Iterate over the input and fill the output collection
  for (HFPreRecHitCollection::const_iterator it = preRecHits->begin(); it != preRecHits->end(); ++it) {
    // The check whether this PMT is single-anode one should go here.
    // Fix this piece of code if/when mixed-anode readout configurations
    // become available.
    const bool thisIsSingleAnodePMT = false;

    // Check if the anodes were tagged bad in the database
    bool taggedBadByDb[2] = {false, false};
    if (useChannelQualityFromDB_) {
      if (checkChannelQualityForDepth3and4_ && !thisIsSingleAnodePMT) {
        HcalDetId anodeIds[2];
        anodeIds[0] = it->id();
        anodeIds[1] = anodeIds[0].secondAnodeId();
        for (unsigned i = 0; i < 2; ++i) {
          const HcalChannelStatus* mydigistatus = myqual->getValues(anodeIds[i].rawId());
          taggedBadByDb[i] = mySeverity->dropChannel(mydigistatus->getValue());
        }
      } else {
        const HcalChannelStatus* mydigistatus = myqual->getValues(it->id().rawId());
        const bool b = mySeverity->dropChannel(mydigistatus->getValue());
        taggedBadByDb[0] = b;
        taggedBadByDb[1] = b;
      }
    }

    // Reconstruct the rechit
    const HFRecHit& rh =
        reco_->reconstruct(*it, conditions->getHcalCalibrations(it->id()), taggedBadByDb, thisIsSingleAnodePMT);

    // The rechit will have the id of 0 if the algorithm
    // decides that it should be dropped
    if (rh.id().rawId())
      rec->push_back(rh);
  }

  // At this point, the rechits contain energy, timing,
  // as well as proper auxiliary words. We do still need
  // to set certain flags using the noise cleanup algoritms.

  // The following flags require the full set of rechits.
  // These flags need to be set consecutively.
  if (setNoiseFlags_) {
    // Step 1:  Set PET flag  (short fibers of |ieta|==29)
    for (HFRecHitCollection::iterator i = rec->begin(); i != rec->end(); ++i) {
      int depth = i->id().depth();
      int ieta = i->id().ieta();
      // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
      if (depth == 2 || abs(ieta) == 29)
        hfPET_->HFSetFlagFromPET(*i, *rec, myqual, mySeverity);
    }

    // Step 2:  Set S8S1 flag (short fibers or |ieta|==29)
    for (HFRecHitCollection::iterator i = rec->begin(); i != rec->end(); ++i) {
      int depth = i->id().depth();
      int ieta = i->id().ieta();
      // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
      if (depth == 2 || abs(ieta) == 29)
        hfS8S1_->HFSetFlagFromS9S1(*i, *rec, myqual, mySeverity);
    }

    // Step 3:  Set S9S1 flag (long fibers)
    for (HFRecHitCollection::iterator i = rec->begin(); i != rec->end(); ++i) {
      int depth = i->id().depth();
      int ieta = i->id().ieta();
      // Short fibers and all channels at |ieta|=29 use PET settings in Algo 3
      if (depth == 1 && abs(ieta) != 29)
        hfS9S1_->HFSetFlagFromS9S1(*i, *rec, myqual, mySeverity);
    }

    // Step 4:  Run HFStripFilter if requested
    if (runHFStripFilter_)
      hfStripFilter_->runFilter(*rec, myqual);
  }

  // Add the output collection to the event record
  e.put(std::move(rec));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HFPhase1Reconstructor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("inputLabel", edm::InputTag("hfprereco"))
      ->setComment("Label for the input HFPreRecHitCollection");
  desc.add<std::string>("algoConfigClass", "HFPhase1PMTParams")
      ->setComment("reconstruction algorithm data to fetch from DB, if any");
  desc.add<bool>("useChannelQualityFromDB", true)
      ->setComment("change the following to True in order to use the channel status from the DB");
  desc.add<bool>("checkChannelQualityForDepth3and4", true);
  desc.add<edm::ParameterSetDescription>("algorithm", fillDescriptionForParseHFPhase1AlgoDescription())
      ->setComment("configure the reconstruction algorithm");

  desc.ifValue(
      edm::ParameterDescription<bool>("runHFStripFilter", true, true),
      false >> edm::EmptyGroupDescription() or true >> edm::ParameterDescription<edm::ParameterSetDescription>(
                                                           "HFStripFilter", HFStripFilter::fillDescription(), true));

  {
    // Define common vectors
    std::vector<double> slopes_S9S1_run1 = {-99999,
                                            0.0164905,
                                            0.0238698,
                                            0.0321383,
                                            0.041296,
                                            0.0513428,
                                            0.0622789,
                                            0.0741041,
                                            0.0868186,
                                            0.100422,
                                            0.135313,
                                            0.136289,
                                            0.0589927};
    std::vector<double> coeffs = {1.0, 2.5, 2.2, 2.0, 1.8, 1.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> slopes_S9S1_run2(slopes_S9S1_run1.size());
    for (size_t i = 0; i < slopes_S9S1_run1.size(); ++i) {
      slopes_S9S1_run2[i] = slopes_S9S1_run1[i] * coeffs[i];
    }

    // S9S1stat configuration
    edm::ParameterSetDescription S9S1statDesc;
    S9S1statDesc.add<std::vector<double>>("short_optimumSlope", slopes_S9S1_run2);
    S9S1statDesc.add<std::vector<double>>(
        "shortEnergyParams",
        {35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093});
    S9S1statDesc.add<std::vector<double>>("shortETParams", std::vector<double>(13, 0));
    S9S1statDesc.add<std::vector<double>>("long_optimumSlope", slopes_S9S1_run2);
    S9S1statDesc.add<std::vector<double>>(
        "longEnergyParams", {43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62});
    S9S1statDesc.add<std::vector<double>>("longETParams", std::vector<double>(13, 0));
    S9S1statDesc.add<int>("HcalAcceptSeverityLevel", 9);
    S9S1statDesc.add<bool>("isS8S1", false);

    // S8S1stat configuration
    edm::ParameterSetDescription S8S1statDesc;
    S8S1statDesc.add<std::vector<double>>(
        "short_optimumSlope", {0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10});
    S8S1statDesc.add<std::vector<double>>("shortEnergyParams",
                                          {40, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100});
    S8S1statDesc.add<std::vector<double>>("shortETParams", std::vector<double>(13, 0));
    S8S1statDesc.add<std::vector<double>>(
        "long_optimumSlope", {0.30, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10});
    S8S1statDesc.add<std::vector<double>>("longEnergyParams",
                                          {40, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100});
    S8S1statDesc.add<std::vector<double>>("longETParams", std::vector<double>(13, 0));
    S8S1statDesc.add<int>("HcalAcceptSeverityLevel", 9);
    S8S1statDesc.add<bool>("isS8S1", true);

    // PETstat configuration
    edm::ParameterSetDescription PETstatDesc;
    PETstatDesc.add<std::vector<double>>("short_R", {0.8});
    PETstatDesc.add<std::vector<double>>(
        "shortEnergyParams",
        {35.1773, 35.37, 35.7933, 36.4472, 37.3317, 38.4468, 39.7925, 41.3688, 43.1757, 45.2132, 47.4813, 49.98, 52.7093});
    PETstatDesc.add<std::vector<double>>("shortETParams", std::vector<double>(13, 0));
    PETstatDesc.add<std::vector<double>>("long_R", {0.98});
    PETstatDesc.add<std::vector<double>>(
        "longEnergyParams", {43.5, 45.7, 48.32, 51.36, 54.82, 58.7, 63.0, 67.72, 72.86, 78.42, 84.4, 90.8, 97.62});
    PETstatDesc.add<std::vector<double>>("longETParams", std::vector<double>(13, 0));
    PETstatDesc.add<std::vector<double>>("short_R_29", {0.8});
    PETstatDesc.add<std::vector<double>>("long_R_29", {0.8});
    PETstatDesc.add<int>("HcalAcceptSeverityLevel", 9);

    // Conditionally add S9S1stat if setNoiseFlags is true
    desc.ifValue(
        edm::ParameterDescription<bool>("setNoiseFlags", true, true),
        false >> edm::EmptyGroupDescription() or
            true >> (edm::ParameterDescription<edm::ParameterSetDescription>("S9S1stat", S9S1statDesc, true) and
                     edm::ParameterDescription<edm::ParameterSetDescription>("S8S1stat", S8S1statDesc, true) and
                     edm::ParameterDescription<edm::ParameterSetDescription>("PETstat", PETstatDesc, true)));
  }

  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HFPhase1Reconstructor);
