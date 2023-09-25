/** \class EcalRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
 *
 **/

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalCleaningAlgo.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerFactory.h"

class EcalRecHitProducer : public edm::stream::EDProducer<> {
public:
  explicit EcalRecHitProducer(const edm::ParameterSet& ps);
  ~EcalRecHitProducer() override;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const bool doEB_;  // consume and use the EB uncalibrated RecHits. An EB collection is produced even if this is false
  const bool doEE_;  // consume and use the EE uncalibrated RecHits. An EE collection is produced even if this is false
  const bool recoverEBIsolatedChannels_;
  const bool recoverEEIsolatedChannels_;
  const bool recoverEBVFE_;
  const bool recoverEEVFE_;
  const bool recoverEBFE_;
  const bool recoverEEFE_;
  const bool killDeadChannels_;

  std::unique_ptr<EcalRecHitWorkerBaseClass> worker_;
  std::unique_ptr<EcalRecHitWorkerBaseClass> workerRecover_;

  std::unique_ptr<EcalCleaningAlgo> cleaningAlgo_;

  edm::EDGetTokenT<EBUncalibratedRecHitCollection> ebUncalibRecHitToken_;
  edm::EDGetTokenT<EEUncalibratedRecHitCollection> eeUncalibRecHitToken_;
  edm::EDGetTokenT<std::set<EBDetId>> ebDetIdToBeRecoveredToken_;
  edm::EDGetTokenT<std::set<EEDetId>> eeDetIdToBeRecoveredToken_;
  edm::EDGetTokenT<std::set<EcalTrigTowerDetId>> ebFEToBeRecoveredToken_;
  edm::EDGetTokenT<std::set<EcalScDetId>> eeFEToBeRecoveredToken_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> ecalChannelStatusToken_;
  const edm::EDPutTokenT<EBRecHitCollection> ebRecHitToken_;
  const edm::EDPutTokenT<EERecHitCollection> eeRecHitToken_;
};

EcalRecHitProducer::EcalRecHitProducer(const edm::ParameterSet& ps)
    : doEB_(!ps.getParameter<edm::InputTag>("EBuncalibRecHitCollection").label().empty()),
      doEE_(!ps.getParameter<edm::InputTag>("EEuncalibRecHitCollection").label().empty()),
      recoverEBIsolatedChannels_(ps.getParameter<bool>("recoverEBIsolatedChannels")),
      recoverEEIsolatedChannels_(ps.getParameter<bool>("recoverEEIsolatedChannels")),
      recoverEBVFE_(ps.getParameter<bool>("recoverEBVFE")),
      recoverEEVFE_(ps.getParameter<bool>("recoverEEVFE")),
      recoverEBFE_(ps.getParameter<bool>("recoverEBFE")),
      recoverEEFE_(ps.getParameter<bool>("recoverEEFE")),
      killDeadChannels_(ps.getParameter<bool>("killDeadChannels")),
      ebRecHitToken_(produces<EBRecHitCollection>(ps.getParameter<std::string>("EBrechitCollection"))),
      eeRecHitToken_(produces<EERecHitCollection>(ps.getParameter<std::string>("EErechitCollection"))) {
  if (doEB_) {
    ebUncalibRecHitToken_ =
        consumes<EBUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("EBuncalibRecHitCollection"));

    if (recoverEBIsolatedChannels_ || recoverEBFE_ || killDeadChannels_) {
      ebDetIdToBeRecoveredToken_ = consumes<std::set<EBDetId>>(ps.getParameter<edm::InputTag>("ebDetIdToBeRecovered"));
    }

    if (recoverEBFE_ || killDeadChannels_) {
      ebFEToBeRecoveredToken_ =
          consumes<std::set<EcalTrigTowerDetId>>(ps.getParameter<edm::InputTag>("ebFEToBeRecovered"));
    }
  }

  if (doEE_) {
    eeUncalibRecHitToken_ =
        consumes<EEUncalibratedRecHitCollection>(ps.getParameter<edm::InputTag>("EEuncalibRecHitCollection"));

    if (recoverEEIsolatedChannels_ || recoverEEFE_ || killDeadChannels_) {
      eeDetIdToBeRecoveredToken_ = consumes<std::set<EEDetId>>(ps.getParameter<edm::InputTag>("eeDetIdToBeRecovered"));
    }

    if (recoverEEFE_ || killDeadChannels_) {
      eeFEToBeRecoveredToken_ = consumes<std::set<EcalScDetId>>(ps.getParameter<edm::InputTag>("eeFEToBeRecovered"));
    }
  }

  if (recoverEBIsolatedChannels_ || recoverEBFE_ || recoverEEIsolatedChannels_ || recoverEEFE_ || killDeadChannels_) {
    ecalChannelStatusToken_ = esConsumes<EcalChannelStatus, EcalChannelStatusRcd>();
  }

  std::string componentType = ps.getParameter<std::string>("algo");
  edm::ConsumesCollector c{consumesCollector()};
  worker_ = EcalRecHitWorkerFactory::get()->create(componentType, ps, c);

  // to recover problematic channels
  componentType = ps.getParameter<std::string>("algoRecover");
  workerRecover_ = EcalRecHitWorkerFactory::get()->create(componentType, ps, c);

  edm::ParameterSet cleaningPs = ps.getParameter<edm::ParameterSet>("cleaningConfig");
  cleaningAlgo_ = std::make_unique<EcalCleaningAlgo>(cleaningPs);
}

EcalRecHitProducer::~EcalRecHitProducer() = default;

void EcalRecHitProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  using namespace edm;

  // collection of rechits to put in the event
  auto ebRecHits = std::make_unique<EBRecHitCollection>();
  auto eeRecHits = std::make_unique<EERecHitCollection>();

  worker_->set(es);

  if (recoverEBIsolatedChannels_ || recoverEEIsolatedChannels_ || recoverEBFE_ || recoverEEFE_ || recoverEBVFE_ ||
      recoverEEVFE_ || killDeadChannels_) {
    workerRecover_->set(es);
  }

  // Make EB rechits
  if (doEB_) {
    const auto& ebUncalibRecHits = evt.get(ebUncalibRecHitToken_);
    LogDebug("EcalRecHitDebug") << "total # EB uncalibrated rechits: " << ebUncalibRecHits.size();

    // loop over uncalibrated rechits to make calibrated ones
    for (const auto& uncalibRecHit : ebUncalibRecHits) {
      worker_->run(evt, uncalibRecHit, *ebRecHits);
    }
  }

  // Make EE rechits
  if (doEE_) {
    const auto& eeUncalibRecHits = evt.get(eeUncalibRecHitToken_);
    LogDebug("EcalRecHitDebug") << "total # EE uncalibrated rechits: " << eeUncalibRecHits.size();

    // loop over uncalibrated rechits to make calibrated ones
    for (const auto& uncalibRecHit : eeUncalibRecHits) {
      worker_->run(evt, uncalibRecHit, *eeRecHits);
    }
  }

  // sort collections before attempting recovery, to avoid insertion of double recHits
  ebRecHits->sort();
  eeRecHits->sort();

  if (recoverEBIsolatedChannels_ || recoverEBFE_ || killDeadChannels_) {
    const auto& detIds = evt.get(ebDetIdToBeRecoveredToken_);
    const auto& chStatus = es.getData(ecalChannelStatusToken_);

    for (const auto& detId : detIds) {
      // get channel status map to treat dead VFE separately
      EcalChannelStatusMap::const_iterator chit = chStatus.find(detId);
      EcalChannelStatusCode chStatusCode;
      if (chit != chStatus.end()) {
        chStatusCode = *chit;
      } else {
        edm::LogError("EcalRecHitProducerError") << "No channel status found for xtal " << detId.rawId()
                                                 << "! something wrong with EcalChannelStatus in your DB? ";
      }
      EcalUncalibratedRecHit urh;
      if (chStatusCode.getStatusCode() == EcalChannelStatusCode::kDeadVFE) {  // dead VFE (from DB info)
        // uses the EcalUncalibratedRecHit to pass the DetId info
        urh = EcalUncalibratedRecHit(detId, 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EB_VFE);
        if (recoverEBVFE_ || killDeadChannels_)
          workerRecover_->run(evt, urh, *ebRecHits);
      } else {
        // uses the EcalUncalibratedRecHit to pass the DetId info
        urh = EcalUncalibratedRecHit(detId, 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EB_single);
        if (recoverEBIsolatedChannels_ || killDeadChannels_)
          workerRecover_->run(evt, urh, *ebRecHits);
      }
    }
  }

  if (recoverEEIsolatedChannels_ || recoverEEVFE_ || killDeadChannels_) {
    const auto& detIds = evt.get(eeDetIdToBeRecoveredToken_);
    const auto& chStatus = es.getData(ecalChannelStatusToken_);

    for (const auto& detId : detIds) {
      // get channel status map to treat dead VFE separately
      EcalChannelStatusMap::const_iterator chit = chStatus.find(detId);
      EcalChannelStatusCode chStatusCode;
      if (chit != chStatus.end()) {
        chStatusCode = *chit;
      } else {
        edm::LogError("EcalRecHitProducerError") << "No channel status found for xtal " << detId.rawId()
                                                 << "! something wrong with EcalChannelStatus in your DB? ";
      }
      EcalUncalibratedRecHit urh;
      if (chStatusCode.getStatusCode() == EcalChannelStatusCode::kDeadVFE) {  // dead VFE (from DB info)
        // uses the EcalUncalibratedRecHit to pass the DetId info
        urh = EcalUncalibratedRecHit(detId, 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EE_VFE);
        if (recoverEEVFE_ || killDeadChannels_)
          workerRecover_->run(evt, urh, *eeRecHits);
      } else {
        // uses the EcalUncalibratedRecHit to pass the DetId info
        urh = EcalUncalibratedRecHit(detId, 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EE_single);
        if (recoverEEIsolatedChannels_ || killDeadChannels_)
          workerRecover_->run(evt, urh, *eeRecHits);
      }
    }
  }

  if (recoverEBFE_ || killDeadChannels_) {
    const auto& ttIds = evt.get(ebFEToBeRecoveredToken_);

    for (const auto& ttId : ttIds) {
      // uses the EcalUncalibratedRecHit to pass the DetId info
      int ieta = ((ttId.ietaAbs() - 1) * 5 + 1) * ttId.zside();  // from EcalTrigTowerConstituentsMap
      int iphi = ((ttId.iphi() - 1) * 5 + 11) % 360;             // from EcalTrigTowerConstituentsMap
      if (iphi <= 0)
        iphi += 360;  // from EcalTrigTowerConstituentsMap
      EcalUncalibratedRecHit urh(
          EBDetId(ieta, iphi, EBDetId::ETAPHIMODE), 0, 0, 0, 0, EcalRecHitWorkerBaseClass::EB_FE);
      workerRecover_->run(evt, urh, *ebRecHits);
    }
  }

  if (recoverEEFE_ || killDeadChannels_) {
    const auto& scIds = evt.get(eeFEToBeRecoveredToken_);

    for (const auto& scId : scIds) {
      // uses the EcalUncalibratedRecHit to pass the DetId info
      if (EEDetId::validDetId((scId.ix() - 1) * 5 + 1, (scId.iy() - 1) * 5 + 1, scId.zside())) {
        EcalUncalibratedRecHit urh(EEDetId((scId.ix() - 1) * 5 + 1, (scId.iy() - 1) * 5 + 1, scId.zside()),
                                   0,
                                   0,
                                   0,
                                   0,
                                   EcalRecHitWorkerBaseClass::EE_FE);
        workerRecover_->run(evt, urh, *eeRecHits);
      }
    }
  }

  // without re-sorting, find (used below in cleaning) will lead
  // to undefined results
  ebRecHits->sort();
  eeRecHits->sort();

  // apply spike cleaning
  if (cleaningAlgo_) {
    cleaningAlgo_->setFlags(*ebRecHits);
    cleaningAlgo_->setFlags(*eeRecHits);
  }

  // put the collection of reconstructed hits in the event
  LogInfo("EcalRecHitInfo") << "total # EB calibrated rechits: " << ebRecHits->size();
  LogInfo("EcalRecHitInfo") << "total # EE calibrated rechits: " << eeRecHits->size();

  evt.put(ebRecHitToken_, std::move(ebRecHits));
  evt.put(eeRecHitToken_, std::move(eeRecHits));
}

void EcalRecHitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("recoverEEVFE", false);
  desc.add<std::string>("EErechitCollection", "EcalRecHitsEE");
  desc.add<bool>("recoverEBIsolatedChannels", false);
  desc.add<bool>("recoverEBVFE", false);
  desc.add<bool>("laserCorrection", true);
  desc.add<double>("EBLaserMIN", 0.5);
  desc.add<bool>("killDeadChannels", true);
  {
    std::vector<int> temp1;
    temp1.reserve(3);
    temp1.push_back(14);
    temp1.push_back(78);
    temp1.push_back(142);
    desc.add<std::vector<int>>("dbStatusToBeExcludedEB", temp1);
  }
  desc.add<edm::InputTag>("EEuncalibRecHitCollection",
                          edm::InputTag("ecalMultiFitUncalibRecHit", "EcalUncalibRecHitsEE"));
  {
    std::vector<int> temp1;
    temp1.reserve(3);
    temp1.push_back(14);
    temp1.push_back(78);
    temp1.push_back(142);
    desc.add<std::vector<int>>("dbStatusToBeExcludedEE", temp1);
  }
  desc.add<double>("EELaserMIN", 0.5);
  desc.add<edm::InputTag>("ebFEToBeRecovered", edm::InputTag("ecalDetIdToBeRecovered", "ebFE"));
  {
    edm::ParameterSetDescription psd0;
    psd0.add<double>("e6e2thresh", 0.04);
    psd0.add<double>("tightenCrack_e6e2_double", 3);
    psd0.add<double>("e4e1Threshold_endcap", 0.3);
    psd0.add<double>("tightenCrack_e4e1_single", 3);
    psd0.add<double>("tightenCrack_e1_double", 2);
    psd0.add<double>("cThreshold_barrel", 4);
    psd0.add<double>("e4e1Threshold_barrel", 0.08);
    psd0.add<double>("tightenCrack_e1_single", 2);
    psd0.add<double>("e4e1_b_barrel", -0.024);
    psd0.add<double>("e4e1_a_barrel", 0.04);
    psd0.add<double>("ignoreOutOfTimeThresh", 1000000000.0);
    psd0.add<double>("cThreshold_endcap", 15);
    psd0.add<double>("e4e1_b_endcap", -0.0125);
    psd0.add<double>("e4e1_a_endcap", 0.02);
    psd0.add<double>("cThreshold_double", 10);
    desc.add<edm::ParameterSetDescription>("cleaningConfig", psd0);
  }
  desc.add<double>("logWarningEtThreshold_EE_FE", 50);
  desc.add<edm::InputTag>("eeDetIdToBeRecovered", edm::InputTag("ecalDetIdToBeRecovered", "eeDetId"));
  desc.add<bool>("recoverEBFE", true);
  desc.add<edm::InputTag>("eeFEToBeRecovered", edm::InputTag("ecalDetIdToBeRecovered", "eeFE"));
  desc.add<edm::InputTag>("ebDetIdToBeRecovered", edm::InputTag("ecalDetIdToBeRecovered", "ebDetId"));
  desc.add<double>("singleChannelRecoveryThreshold", 8);
  desc.add<double>("sum8ChannelRecoveryThreshold", 0.);
  desc.add<edm::FileInPath>("bdtWeightFileNoCracks",
                            edm::FileInPath("RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/BDTWeights/"
                                            "bdtgAllRH_8GT700MeV_noCracks_ZskimData2017_v1.xml"));
  desc.add<edm::FileInPath>("bdtWeightFileCracks",
                            edm::FileInPath("RecoLocalCalo/EcalDeadChannelRecoveryAlgos/data/BDTWeights/"
                                            "bdtgAllRH_8GT700MeV_onlyCracks_ZskimData2017_v1.xml"));
  {
    std::vector<std::string> temp1;
    temp1.reserve(9);
    temp1.push_back("kNoisy");
    temp1.push_back("kNNoisy");
    temp1.push_back("kFixedG6");
    temp1.push_back("kFixedG1");
    temp1.push_back("kFixedG0");
    temp1.push_back("kNonRespondingIsolated");
    temp1.push_back("kDeadVFE");
    temp1.push_back("kDeadFE");
    temp1.push_back("kNoDataNoTP");
    desc.add<std::vector<std::string>>("ChannelStatusToBeExcluded", temp1);
  }
  desc.add<std::string>("EBrechitCollection", "EcalRecHitsEB");
  desc.add<edm::InputTag>("triggerPrimitiveDigiCollection", edm::InputTag("ecalDigis", "EcalTriggerPrimitives"));
  desc.add<bool>("recoverEEFE", true);
  desc.add<std::string>("singleChannelRecoveryMethod", "NeuralNetworks");
  desc.add<double>("EBLaserMAX", 3.0);
  {
    edm::ParameterSetDescription psd0;
    {
      std::vector<std::string> temp2;
      temp2.reserve(4);
      temp2.push_back("kOk");
      temp2.push_back("kDAC");
      temp2.push_back("kNoLaser");
      temp2.push_back("kNoisy");
      psd0.add<std::vector<std::string>>("kGood", temp2);
    }
    {
      std::vector<std::string> temp2;
      temp2.reserve(3);
      temp2.push_back("kFixedG0");
      temp2.push_back("kNonRespondingIsolated");
      temp2.push_back("kDeadVFE");
      psd0.add<std::vector<std::string>>("kNeighboursRecovered", temp2);
    }
    {
      std::vector<std::string> temp2;
      temp2.reserve(1);
      temp2.push_back("kNoDataNoTP");
      psd0.add<std::vector<std::string>>("kDead", temp2);
    }
    {
      std::vector<std::string> temp2;
      temp2.reserve(3);
      temp2.push_back("kNNoisy");
      temp2.push_back("kFixedG6");
      temp2.push_back("kFixedG1");
      psd0.add<std::vector<std::string>>("kNoisy", temp2);
    }
    {
      std::vector<std::string> temp2;
      temp2.reserve(1);
      temp2.push_back("kDeadFE");
      psd0.add<std::vector<std::string>>("kTowerRecovered", temp2);
    }
    desc.add<edm::ParameterSetDescription>("flagsMapDBReco", psd0);
  }
  desc.add<edm::InputTag>("EBuncalibRecHitCollection",
                          edm::InputTag("ecalMultiFitUncalibRecHit", "EcalUncalibRecHitsEB"));
  desc.add<std::string>("algoRecover", "EcalRecHitWorkerRecover");
  desc.add<std::string>("algo", "EcalRecHitWorkerSimple");
  desc.add<double>("EELaserMAX", 8.0);
  desc.add<double>("logWarningEtThreshold_EB_FE", 50);
  desc.add<bool>("recoverEEIsolatedChannels", false);
  desc.add<edm::ESInputTag>("timeCalibTag", edm::ESInputTag());
  desc.add<edm::ESInputTag>("timeOffsetTag", edm::ESInputTag());
  desc.add<bool>("skipTimeCalib", false);
  descriptions.add("ecalRecHit", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalRecHitProducer);
