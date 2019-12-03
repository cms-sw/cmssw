#ifndef RecoParticleFlow_PFClusterProducer_PFEcalRecHitQTests_h
#define RecoParticleFlow_PFClusterProducer_PFEcalRecHitQTests_h

#include <memory>
#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitQTestBase.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalPFSeedingThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFSeedingThresholdsRcd.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include <iostream>

//
//  Quality test that checks threshold
//
class PFRecHitQTestThreshold : public PFRecHitQTestBase {
public:
  PFRecHitQTestThreshold() {}

  PFRecHitQTestThreshold(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig), threshold_(iConfig.getParameter<double>("threshold")) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {}

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override {
    return fullReadOut or pass(hit);
  }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override { return pass(hit); }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override { return pass(hit); }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override { return pass(hit); }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return pass(hit); }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return pass(hit); }

protected:
  double threshold_;

  bool pass(const reco::PFRecHit& hit) { return hit.energy() > threshold_; }
};

//
//  Quality test that checks threshold read from the DB
//
class PFRecHitQTestDBThreshold : public PFRecHitQTestBase {
public:
  PFRecHitQTestDBThreshold() : eventSetup_(nullptr) {}

  PFRecHitQTestDBThreshold(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig),
        applySelectionsToAllCrystals_(iConfig.getParameter<bool>("applySelectionsToAllCrystals")),
        eventSetup_(nullptr) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override { eventSetup_ = &iSetup; }

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override {
    if (applySelectionsToAllCrystals_)
      return pass(hit);
    return fullReadOut or pass(hit);
  }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override { return pass(hit); }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override { return pass(hit); }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override { return pass(hit); }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return pass(hit); }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return pass(hit); }

protected:
  bool applySelectionsToAllCrystals_;
  const edm::EventSetup* eventSetup_;

  bool pass(const reco::PFRecHit& hit) {
    edm::ESHandle<EcalPFRecHitThresholds> ths;
    (*eventSetup_).get<EcalPFRecHitThresholdsRcd>().get(ths);

    float threshold = (*ths)[hit.detId()];
    return hit.energy() > threshold;
  }
};

//
//  Quality test that checks kHCAL Severity
//
class PFRecHitQTestHCALChannel : public PFRecHitQTestBase {
public:
  PFRecHitQTestHCALChannel() {}

  PFRecHitQTestHCALChannel(const edm::ParameterSet& iConfig) : PFRecHitQTestBase(iConfig) {
    thresholds_ = iConfig.getParameter<std::vector<int> >("maxSeverities");
    cleanThresholds_ = iConfig.getParameter<std::vector<double> >("cleaningThresholds");
    std::vector<std::string> flags = iConfig.getParameter<std::vector<std::string> >("flags");
    for (auto& flag : flags) {
      if (flag == "Standard") {
        flags_.push_back(-1);
        depths_.push_back(-1);
      } else if (flag == "HFInTime") {
        flags_.push_back(1 << HcalCaloFlagLabels::HFInTimeWindow);
        depths_.push_back(-1);
      } else if (flag == "HFDigi") {
        flags_.push_back(1 << HcalCaloFlagLabels::HFDigiTime);
        depths_.push_back(-1);
      } else if (flag == "HFLong") {
        flags_.push_back(1 << HcalCaloFlagLabels::HFLongShort);
        depths_.push_back(1);
      } else if (flag == "HFShort") {
        flags_.push_back(1 << HcalCaloFlagLabels::HFLongShort);
        depths_.push_back(2);
      } else {
        flags_.push_back(-1);
        depths_.push_back(-1);
      }
    }
  }

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {
    edm::ESHandle<HcalTopology> topo;
    iSetup.get<HcalRecNumberingRecord>().get(topo);
    theHcalTopology_ = topo.product();
    edm::ESHandle<HcalChannelQuality> hcalChStatus;
    iSetup.get<HcalChannelQualityRcd>().get("withTopo", hcalChStatus);
    theHcalChStatus_ = hcalChStatus.product();
    edm::ESHandle<HcalSeverityLevelComputer> hcalSevLvlComputerHndl;
    iSetup.get<HcalSeverityLevelComputerRcd>().get(hcalSevLvlComputerHndl);
    hcalSevLvlComputer_ = hcalSevLvlComputerHndl.product();
  }

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override { return true; }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override {
    return test(rh.detid(), rh.energy(), rh.flags(), clean);
  }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override {
    return test(rh.detid(), rh.energy(), rh.flags(), clean);
  }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override {
    return test(rh.detid(), rh.energy(), rh.flags(), clean);
  }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return true; }

protected:
  std::vector<int> thresholds_;
  std::vector<double> cleanThresholds_;
  std::vector<int> flags_;
  std::vector<int> depths_;
  const HcalTopology* theHcalTopology_;
  const HcalChannelQuality* theHcalChStatus_;
  const HcalSeverityLevelComputer* hcalSevLvlComputer_;

  bool test(unsigned aDETID, double energy, int flags, bool& clean) {
    HcalDetId detid = (HcalDetId)aDETID;
    if (theHcalTopology_->getMergePositionFlag() and detid.subdet() == HcalEndcap) {
      detid = theHcalTopology_->idFront(detid);
    }

    const HcalChannelStatus* theStatus = theHcalChStatus_->getValues(detid);
    unsigned theStatusValue = theStatus->getValue();
    // Now get severity of problems for the given detID, based on the rechit flag word and the channel quality status value
    for (unsigned int i = 0; i < thresholds_.size(); ++i) {
      int hitSeverity = 0;
      if (energy < cleanThresholds_[i])
        continue;

      if (flags_[i] < 0) {
        hitSeverity = hcalSevLvlComputer_->getSeverityLevel(detid, flags, theStatusValue);
      } else {
        hitSeverity = hcalSevLvlComputer_->getSeverityLevel(detid, flags & flags_[i], theStatusValue);
      }

      if (hitSeverity > thresholds_[i] and ((depths_[i] < 0 or (depths_[i] == detid.depth())))) {
        clean = true;
        return false;
      }
    }
    return true;
  }
};

//
//  Quality test that applies threshold and timing as a function of depth
//
class PFRecHitQTestHCALTimeVsDepth : public PFRecHitQTestBase {
public:
  PFRecHitQTestHCALTimeVsDepth() {}

  PFRecHitQTestHCALTimeVsDepth(const edm::ParameterSet& iConfig) : PFRecHitQTestBase(iConfig) {
    std::vector<edm::ParameterSet> psets = iConfig.getParameter<std::vector<edm::ParameterSet> >("cuts");
    for (auto& pset : psets) {
      depths_.push_back(pset.getParameter<int>("depth"));
      minTimes_.push_back(pset.getParameter<double>("minTime"));
      maxTimes_.push_back(pset.getParameter<double>("maxTime"));
      thresholds_.push_back(pset.getParameter<double>("threshold"));
    }
  }

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {}

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override { return true; }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override {
    return test(rh.detid(), rh.energy(), rh.time(), clean);
  }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override {
    return test(rh.detid(), rh.energy(), rh.time(), clean);
  }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override {
    return test(rh.detid(), rh.energy(), rh.time(), clean);
  }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return true; }

protected:
  std::vector<int> depths_;
  std::vector<double> minTimes_;
  std::vector<double> maxTimes_;
  std::vector<double> thresholds_;

  bool test(unsigned aDETID, double energy, double time, bool& clean) {
    HcalDetId detid(aDETID);
    for (unsigned int i = 0; i < depths_.size(); ++i) {
      if (detid.depth() == depths_[i]) {
        if ((time < minTimes_[i] or time > maxTimes_[i]) and energy > thresholds_[i]) {
          clean = true;
          return false;
        }
        break;
      }
    }
    return true;
  }
};

//
//  Quality test that applies threshold  as a function of depth
//
class PFRecHitQTestHCALThresholdVsDepth : public PFRecHitQTestBase {
public:
  PFRecHitQTestHCALThresholdVsDepth() {}

  PFRecHitQTestHCALThresholdVsDepth(const edm::ParameterSet& iConfig) : PFRecHitQTestBase(iConfig) {
    std::vector<edm::ParameterSet> psets = iConfig.getParameter<std::vector<edm::ParameterSet> >("cuts");
    for (auto& pset : psets) {
      depths_ = pset.getParameter<std::vector<int> >("depth");
      thresholds_ = pset.getParameter<std::vector<double> >("threshold");
      detector_ = pset.getParameter<int>("detectorEnum");
      if (thresholds_.size() != depths_.size()) {
        throw cms::Exception("InvalidPFRecHitThreshold") << "PFRecHitThreshold mismatch with the numbers of depths";
      }
    }
  }

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {}

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override { return true; }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override {
    return test(rh.detid(), rh.energy(), rh.time(), clean);
  }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override {
    return test(rh.detid(), rh.energy(), rh.time(), clean);
  }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override {
    return test(rh.detid(), rh.energy(), rh.time(), clean);
  }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return true; }

protected:
  std::vector<int> depths_;
  std::vector<double> thresholds_;
  int detector_;

  bool test(unsigned aDETID, double energy, double time, bool& clean) {
    HcalDetId detid(aDETID);

    for (unsigned int i = 0; i < thresholds_.size(); ++i) {
      if (detid.depth() == depths_[i] && detid.subdet() == detector_) {
        if (energy < thresholds_[i]) {
          clean = false;
          return false;
        }
        break;
      }
    }
    return true;
  }
};

//
//  Quality test that checks HO threshold applying different threshold in rings
//
class PFRecHitQTestHOThreshold : public PFRecHitQTestBase {
public:
  PFRecHitQTestHOThreshold() : threshold0_(0.), threshold12_(0.) {}

  PFRecHitQTestHOThreshold(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig),
        threshold0_(iConfig.getParameter<double>("threshold_ring0")),
        threshold12_(iConfig.getParameter<double>("threshold_ring12")) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {}

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override { return true; }

  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override {
    HcalDetId detid(rh.detid());
    if (abs(detid.ieta()) <= 4 and hit.energy() > threshold0_)
      return true;
    if (abs(detid.ieta()) > 4 and hit.energy() > threshold12_)
      return true;

    return false;
  }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return true; }

protected:
  const double threshold0_;
  const double threshold12_;
};

//
//  Quality test that checks threshold as a function of ECAL eta-ring
//
#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
class PFRecHitQTestECALMultiThreshold : public PFRecHitQTestBase {
public:
  PFRecHitQTestECALMultiThreshold() {}

  PFRecHitQTestECALMultiThreshold(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig),
        thresholds_(iConfig.getParameter<std::vector<double> >("thresholds")),
        applySelectionsToAllCrystals_(iConfig.getParameter<bool>("applySelectionsToAllCrystals")) {
    if (thresholds_.size() != EcalRingCalibrationTools::N_RING_TOTAL)
      throw edm::Exception(edm::errors::Configuration, "ValueError")
          << "thresholds is expected to have " << EcalRingCalibrationTools::N_RING_TOTAL << " elements but has "
          << thresholds_.size();
  }

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    CaloSubdetectorGeometry const* endcapGeometry = pG->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    endcapGeometrySet_ = false;
    if (endcapGeometry) {
      EcalRingCalibrationTools::setCaloGeometry(&(*pG));
      endcapGeometrySet_ = true;
    }
  }

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override {
    if (applySelectionsToAllCrystals_)
      return pass(hit);
    else
      return fullReadOut or pass(hit);
  }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override { return true; }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return true; }

protected:
  const std::vector<double> thresholds_;
  bool endcapGeometrySet_;

  // apply selections to all crystals
  bool applySelectionsToAllCrystals_;

  bool pass(const reco::PFRecHit& hit) {
    DetId detId(hit.detId());

    // this is to skip endcap ZS for Phase2 until there is a defined geometry
    // apply the loosest ZS threshold, for the first eta-ring in EB
    if (not endcapGeometrySet_) {
      // there is only ECAL EB in Phase 2
      if (detId.subdetId() != EcalBarrel)
        return true;

      //   0-169: EB  eta-rings
      // 170-208: EE- eta rings
      // 209-247: EE+ eta rings
      int firstEBRing = 0;
      return (hit.energy() > thresholds_[firstEBRing]);
    }

    int iring = EcalRingCalibrationTools::getRingIndex(detId);
    if (hit.energy() > thresholds_[iring])
      return true;

    return false;
  }
};

//
//  Quality test that checks ecal quality cuts
//
class PFRecHitQTestECAL : public PFRecHitQTestBase {
public:
  PFRecHitQTestECAL() {}

  PFRecHitQTestECAL(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig),
        thresholdCleaning_(iConfig.getParameter<double>("cleaningThreshold")),
        timingCleaning_(iConfig.getParameter<bool>("timingCleaning")),
        topologicalCleaning_(iConfig.getParameter<bool>("topologicalCleaning")),
        skipTTRecoveredHits_(iConfig.getParameter<bool>("skipTTRecoveredHits")) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {}

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override {
    if (skipTTRecoveredHits_ and rh.checkFlag(EcalRecHit::kTowerRecovered)) {
      clean = true;
      return false;
    }
    if (timingCleaning_ and rh.energy() > thresholdCleaning_ and rh.checkFlag(EcalRecHit::kOutOfTime)) {
      clean = true;
      return false;
    }

    if (topologicalCleaning_ and (rh.checkFlag(EcalRecHit::kWeird) or rh.checkFlag(EcalRecHit::kDiWeird))) {
      clean = true;
      return false;
    }

    return true;
  }

  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return true; }

protected:
  double thresholdCleaning_;
  bool timingCleaning_;
  bool topologicalCleaning_;
  bool skipTTRecoveredHits_;
};

//
//  Quality test that checks ES quality cuts
//
class PFRecHitQTestES : public PFRecHitQTestBase {
public:
  PFRecHitQTestES() : thresholdCleaning_(0.), topologicalCleaning_(false) {}

  PFRecHitQTestES(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig),
        thresholdCleaning_(iConfig.getParameter<double>("cleaningThreshold")),
        topologicalCleaning_(iConfig.getParameter<bool>("topologicalCleaning")) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {}

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override {
    if (rh.energy() < thresholdCleaning_) {
      clean = false;
      return false;
    }

    if (topologicalCleaning_ and
        (rh.checkFlag(EcalRecHit::kESDead) or rh.checkFlag(EcalRecHit::kESTS13Sigmas) or
         rh.checkFlag(EcalRecHit::kESBadRatioFor12) or rh.checkFlag(EcalRecHit::kESBadRatioFor23Upper) or
         rh.checkFlag(EcalRecHit::kESBadRatioFor23Lower) or rh.checkFlag(EcalRecHit::kESTS1Largest) or
         rh.checkFlag(EcalRecHit::kESTS3Largest) or rh.checkFlag(EcalRecHit::kESTS3Negative))) {
      clean = false;
      return false;
    }

    return true;
  }

  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return true; }

protected:
  const double thresholdCleaning_;
  const bool topologicalCleaning_;
};

//
//  Quality test that calibrates tower 29 of HCAL
//
class PFRecHitQTestHCALCalib29 : public PFRecHitQTestBase {
public:
  PFRecHitQTestHCALCalib29() : calibFactor_(0.) {}

  PFRecHitQTestHCALCalib29(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig), calibFactor_(iConfig.getParameter<double>("calibFactor")) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {}

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override { return true; }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override {
    HcalDetId detId(hit.detId());
    if (abs(detId.ieta()) == 29)
      hit.setEnergy(hit.energy() * calibFactor_);
    return true;
  }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override { return true; }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override { return true; }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override {
    CaloTowerDetId detId(hit.detId());
    if (detId.ietaAbs() == 29)
      hit.setEnergy(hit.energy() * calibFactor_);
    return true;
  }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return true; }

protected:
  const float calibFactor_;
};

class PFRecHitQTestThresholdInMIPs : public PFRecHitQTestBase {
public:
  PFRecHitQTestThresholdInMIPs() : recHitEnergy_keV_(false), threshold_(0.), mip_(0.), recHitEnergyMultiplier_(0.) {}

  PFRecHitQTestThresholdInMIPs(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig),
        recHitEnergy_keV_(iConfig.getParameter<bool>("recHitEnergyIs_keV")),
        threshold_(iConfig.getParameter<double>("thresholdInMIPs")),
        mip_(iConfig.getParameter<double>("mipValueInkeV")),
        recHitEnergyMultiplier_(iConfig.getParameter<double>("recHitEnergyMultiplier")) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {}

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override {
    const double newE =
        (recHitEnergy_keV_ ? 1.0e-6 * rh.energy() * recHitEnergyMultiplier_ : rh.energy() * recHitEnergyMultiplier_);
    hit.setEnergy(newE);
    return pass(hit);
  }

protected:
  const bool recHitEnergy_keV_;
  const double threshold_, mip_, recHitEnergyMultiplier_;

  bool pass(const reco::PFRecHit& hit) {
    const double hitValueInMIPs = 1e6 * hit.energy() / mip_;
    return hitValueInMIPs > threshold_;
  }
};

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
class PFRecHitQTestThresholdInThicknessNormalizedMIPs : public PFRecHitQTestBase {
public:
  PFRecHitQTestThresholdInThicknessNormalizedMIPs()
      : geometryInstance_(""), recHitEnergy_keV_(0.), threshold_(0.), mip_(0.), recHitEnergyMultiplier_(0.) {}

  PFRecHitQTestThresholdInThicknessNormalizedMIPs(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig),
        geometryInstance_(iConfig.getParameter<std::string>("geometryInstance")),
        recHitEnergy_keV_(iConfig.getParameter<bool>("recHitEnergyIs_keV")),
        threshold_(iConfig.getParameter<double>("thresholdInMIPs")),
        mip_(iConfig.getParameter<double>("mipValueInkeV")),
        recHitEnergyMultiplier_(iConfig.getParameter<double>("recHitEnergyMultiplier")) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {
    edm::ESHandle<HGCalGeometry> geoHandle;
    iSetup.get<IdealGeometryRecord>().get(geometryInstance_, geoHandle);
    ddd_ = &(geoHandle->topology().dddConstants());
  }

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestThresholdInMIPs only works for HGCAL!";
    return false;
  }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override {
    const double newE =
        (recHitEnergy_keV_ ? 1.0e-6 * rh.energy() * recHitEnergyMultiplier_ : rh.energy() * recHitEnergyMultiplier_);
    const int wafer = HGCalDetId(rh.detid()).wafer();
    const float mult = (float)ddd_->waferTypeL(wafer);  // 1 for 100um, 2 for 200um, 3 for 300um
    hit.setEnergy(newE);
    return pass(hit, mult);
  }

protected:
  const std::string geometryInstance_;
  const bool recHitEnergy_keV_;
  const double threshold_, mip_, recHitEnergyMultiplier_;
  const HGCalDDDConstants* ddd_;

  bool pass(const reco::PFRecHit& hit, const float mult) {
    const double hitValueInMIPs = 1e6 * hit.energy() / (mult * mip_);
    return hitValueInMIPs > threshold_;
  }
};

class PFRecHitQTestHGCalThresholdSNR : public PFRecHitQTestBase {
public:
  PFRecHitQTestHGCalThresholdSNR() : thresholdSNR_(0.) {}

  PFRecHitQTestHGCalThresholdSNR(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig), thresholdSNR_(iConfig.getParameter<double>("thresholdSNR")) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {}

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestHGCalThresholdSNR only works for HGCAL!";
    return false;
  }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestHGCalThresholdSNR only works for HGCAL!";
    return false;
  }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestHGCalThresholdSNR only works for HGCAL!";
    return false;
  }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestHGCalThresholdSNR only works for HGCAL!";
    return false;
  }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override {
    throw cms::Exception("WrongDetector") << "PFRecHitQTestHGCalThresholdSNR only works for HGCAL!";
    return false;
  }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override {
    return rh.signalOverSigmaNoise() >= thresholdSNR_;
  }

protected:
  const double thresholdSNR_;
};

//  M.G. Quality test that checks seeding threshold read from the DB
//
class PFRecHitQTestDBSeedingThreshold : public PFRecHitQTestBase {
public:
  PFRecHitQTestDBSeedingThreshold(const edm::ParameterSet& iConfig)
      : PFRecHitQTestBase(iConfig),
        applySelectionsToAllCrystals_(iConfig.getParameter<bool>("applySelectionsToAllCrystals")) {}

  void beginEvent(const edm::Event& event, const edm::EventSetup& iSetup) override {
    iSetup.get<EcalPFSeedingThresholdsRcd>().get(ths_);
  }

  bool test(reco::PFRecHit& hit, const EcalRecHit& rh, bool& clean, bool fullReadOut) override {
    if (applySelectionsToAllCrystals_)
      return pass(hit);
    return fullReadOut or pass(hit);
  }
  bool test(reco::PFRecHit& hit, const HBHERecHit& rh, bool& clean) override { return pass(hit); }

  bool test(reco::PFRecHit& hit, const HFRecHit& rh, bool& clean) override { return pass(hit); }
  bool test(reco::PFRecHit& hit, const HORecHit& rh, bool& clean) override { return pass(hit); }

  bool test(reco::PFRecHit& hit, const CaloTower& rh, bool& clean) override { return pass(hit); }

  bool test(reco::PFRecHit& hit, const HGCRecHit& rh, bool& clean) override { return pass(hit); }

protected:
  bool applySelectionsToAllCrystals_;
  edm::ESHandle<EcalPFSeedingThresholds> ths_;

  bool pass(const reco::PFRecHit& hit) {
    float threshold = (*ths_)[hit.detId()];
    return (hit.energy() > threshold);
  }
};

#endif
