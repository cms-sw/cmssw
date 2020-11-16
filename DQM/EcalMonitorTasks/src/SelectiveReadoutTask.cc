#include "DQM/EcalMonitorTasks/interface/SelectiveReadoutTask.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "CondFormats/DataRecord/interface/EcalSRSettingsRcd.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

#include "DQM/EcalCommon/interface/FEFlags.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  SelectiveReadoutTask::SelectiveReadoutTask()
      : DQWorkerTask(),
        useCondDb_(false),
        iFirstSample_(0),
        ZSFIRWeights_(nFIRTaps, 0.),
        suppressed_(),
        flags_(nRU, -1) {}

  void SelectiveReadoutTask::setParams(edm::ParameterSet const& _params) {
    useCondDb_ = _params.getUntrackedParameter<bool>("useCondDb");
    iFirstSample_ = _params.getUntrackedParameter<int>("DCCZS1stSample");

    if (!useCondDb_) {
      std::vector<double> normWeights(_params.getUntrackedParameter<std::vector<double> >("ZSFIRWeights"));
      setFIRWeights_(normWeights);
    }
  }

  void SelectiveReadoutTask::addDependencies(DependencySet& _dependencies) {
    _dependencies.push_back(Dependency(kEBDigi, kEcalRawData, kEBSrFlag));
    _dependencies.push_back(Dependency(kEEDigi, kEcalRawData, kEESrFlag));
  }

  void SelectiveReadoutTask::beginRun(edm::Run const&, edm::EventSetup const& _es) {
    using namespace std;

    if (useCondDb_) {
      edm::ESHandle<EcalSRSettings> hSr;
      _es.get<EcalSRSettingsRcd>().get(hSr);

      vector<vector<float> > weights(hSr->dccNormalizedWeights_);
      if (weights.size() == 1) {
        vector<double> normWeights;
        for (vector<float>::iterator it(weights[0].begin()); it != weights[0].end(); it++)
          normWeights.push_back(*it);

        setFIRWeights_(normWeights);
      } else
        edm::LogWarning("EcalDQM") << "SelectiveReadoutTask: DCC weight set is not exactly 1.";
    }
  }

  void SelectiveReadoutTask::beginEvent(edm::Event const&, edm::EventSetup const&, bool const&, bool&) {
    flags_.assign(nRU, -1);
    suppressed_.clear();
  }

  void SelectiveReadoutTask::runOnSource(FEDRawDataCollection const& _fedRaw) {
    MESet& meDCCSize(MEs_.at("DCCSize"));
    MESet& meDCCSizeProf(MEs_.at("DCCSizeProf"));
    MESet& meEventSize(MEs_.at("EventSize"));

    float ebSize(0.), eemSize(0.), eepSize(0.);

    // DCC event size
    for (int iFED(601); iFED <= 654; iFED++) {
      float size(_fedRaw.FEDData(iFED).size() / 1024.);
      meDCCSize.fill(iFED - 600, size);
      meDCCSizeProf.fill(iFED - 600, size);
      if (iFED - 601 <= kEEmHigh)
        eemSize += size;
      else if (iFED - 601 >= kEEpLow)
        eepSize += size;
      else
        ebSize += size;
    }

    meEventSize.fill(-EcalEndcap, eemSize / 9.);
    meEventSize.fill(EcalEndcap, eepSize / 9.);
    meEventSize.fill(EcalBarrel, ebSize / 36.);
  }

  void SelectiveReadoutTask::runOnRawData(EcalRawDataCollection const& _dcchs) {
    for (EcalRawDataCollection::const_iterator dcchItr(_dcchs.begin()); dcchItr != _dcchs.end(); ++dcchItr) {
      std::vector<short> const& feStatus(dcchItr->getFEStatus());
      unsigned nFE(feStatus.size());
      for (unsigned iFE(0); iFE < nFE; ++iFE)
        if (feStatus[iFE] == Disabled)
          suppressed_.insert(std::make_pair(dcchItr->id(), iFE + 1));
    }
  }

  template <typename SRFlagCollection>
  void SelectiveReadoutTask::runOnSrFlags(SRFlagCollection const& _srfs, Collections _col) {
    MESet& meFlagCounterMap(MEs_.at("FlagCounterMap"));
    MESet& meFullReadoutMap(MEs_.at("FullReadoutMap"));
    MESet& meZS1Map(MEs_.at("ZS1Map"));
    MESet& meZSMap(MEs_.at("ZSMap"));
    MESet& meRUForcedMap(MEs_.at("RUForcedMap"));

    double nFR(0.);

    std::for_each(_srfs.begin(), _srfs.end(), [&](typename SRFlagCollection::value_type const& srf) {
      DetId const& id(srf.id());
      int flag(srf.value());

      meFlagCounterMap.fill(id);

      unsigned iRU(-1);
      if (id.subdetId() == EcalTriggerTower)
        iRU = EcalTrigTowerDetId(id).hashedIndex();
      else
        iRU = EcalScDetId(id).hashedIndex() + EcalTrigTowerDetId::kEBTotalTowers;
      flags_[iRU] = flag;

      switch (flag & ~EcalSrFlag::SRF_FORCED_MASK) {
        case EcalSrFlag::SRF_FULL:
          meFullReadoutMap.fill(id);
          nFR += 1.;
          break;
        case EcalSrFlag::SRF_ZS1:
          meZS1Map.fill(id);
          // fallthrough
        case EcalSrFlag::SRF_ZS2:
          meZSMap.fill(id);
          break;
        default:
          break;
      }

      if (flag & EcalSrFlag::SRF_FORCED_MASK)
        meRUForcedMap.fill(id);
    });

    MEs_.at("FullReadout").fill(_col == kEBSrFlag ? EcalBarrel : EcalEndcap, nFR);
  }

  template <typename DigiCollection>
  void SelectiveReadoutTask::runOnDigis(DigiCollection const& _digis, Collections _collection) {
    MESet& meHighIntOutput(MEs_.at("HighIntOutput"));
    MESet& meLowIntOutput(MEs_.at("LowIntOutput"));
    MESet& meHighIntPayload(MEs_.at("HighIntPayload"));
    MESet& meLowIntPayload(MEs_.at("LowIntPayload"));
    MESet& meTowerSize(MEs_.at("TowerSize"));
    MESet& meZSFullReadoutMap(MEs_.at("ZSFullReadoutMap"));
    MESet& meFRDroppedMap(MEs_.at("FRDroppedMap"));
    MESet& meZSFullReadout(MEs_.at("ZSFullReadout"));
    MESet& meFRDropped(MEs_.at("FRDropped"));

    bool isEB(_collection == kEBDigi);

    unsigned const nTower(isEB ? unsigned(EcalTrigTowerDetId::kEBTotalTowers)
                               : unsigned(EcalScDetId::kSizeForDenseIndexing));

    std::vector<unsigned> sizes(nTower, 0);

    int nHighInt[] = {0, 0};
    int nLowInt[] = {0, 0};

    for (typename DigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr) {
      DetId const& id(digiItr->id());

      unsigned iTower(-1);
      unsigned iRU(-1);

      if (isEB) {
        iTower = EBDetId(id).tower().hashedIndex();
        iRU = iTower;
      } else {
        iTower = EEDetId(id).sc().hashedIndex();
        iRU = iTower + EcalTrigTowerDetId::kEBTotalTowers;
      }

      if (flags_[iRU] < 0)
        continue;

      sizes[iTower] += 1;

      // SR filter output calculation

      EcalDataFrame frame(*digiItr);

      int ZSFIRValue(0);  // output

      bool gain12saturated(false);
      const int gain12(0x01);

      for (int iWeight(0); iWeight < nFIRTaps; ++iWeight) {
        int iSample(iFirstSample_ + iWeight - 1);

        if (iSample >= 0 && iSample < frame.size()) {
          EcalMGPASample sample(frame[iSample]);
          if (sample.gainId() != gain12) {
            gain12saturated = true;
            break;
          }
          ZSFIRValue += sample.adc() * ZSFIRWeights_[iWeight];
        } else {
          edm::LogWarning("EcalDQM") << "SelectiveReadoutTask: Not enough samples in data frame or "
                                        "'ecalDccZs1stSample' module parameter is not valid";
        }
      }

      if (gain12saturated)
        ZSFIRValue = std::numeric_limits<int>::max();
      else
        ZSFIRValue /= (0x1 << 8);  //discards the 8 LSBs

      //ZS passed if weighted sum above ZS threshold or if
      //one sample has a lower gain than gain 12 (that is gain 12 output
      //is saturated)

      bool highInterest((flags_[iRU] & ~EcalSrFlag::SRF_FORCED_MASK) == EcalSrFlag::SRF_FULL);

      if (highInterest) {
        meHighIntOutput.fill(id, ZSFIRValue);
        if (isEB || dccId(id) - 1 <= kEEmHigh)
          nHighInt[0] += 1;
        else
          nHighInt[1] += 1;
      } else {
        meLowIntOutput.fill(id, ZSFIRValue);
        if (isEB || dccId(id) - 1 <= kEEmHigh)
          nLowInt[0] += 1;
        else
          nLowInt[1] += 1;
      }
    }

    if (isEB) {
      meHighIntPayload.fill(EcalBarrel, nHighInt[0] * bytesPerCrystal / 1024. / nEBDCC);
      meLowIntPayload.fill(EcalBarrel, nLowInt[0] * bytesPerCrystal / 1024. / nEBDCC);
    } else {
      meHighIntPayload.fill(-EcalEndcap, nHighInt[0] * bytesPerCrystal / 1024. / (nEEDCC / 2));
      meHighIntPayload.fill(EcalEndcap, nHighInt[1] * bytesPerCrystal / 1024. / (nEEDCC / 2));
      meLowIntPayload.fill(-EcalEndcap, nLowInt[0] * bytesPerCrystal / 1024. / (nEEDCC / 2));
      meLowIntPayload.fill(EcalEndcap, nLowInt[1] * bytesPerCrystal / 1024. / (nEEDCC / 2));
    }

    unsigned iRU(isEB ? 0 : EcalTrigTowerDetId::kEBTotalTowers);
    for (unsigned iTower(0); iTower < nTower; ++iTower, ++iRU) {
      DetId id;
      if (isEB)
        id = EcalTrigTowerDetId::detIdFromDenseIndex(iTower);
      else
        id = EcalScDetId::unhashIndex(iTower);

      double towerSize(sizes[iTower] * bytesPerCrystal);

      meTowerSize.fill(id, towerSize);

      if (flags_[iRU] < 0)
        continue;

      int dccid(dccId(id));
      int towerid(towerId(id));

      if (suppressed_.find(std::make_pair(dccid, towerid)) != suppressed_.end())
        continue;

      int flag(flags_[iRU] & ~EcalSrFlag::SRF_FORCED_MASK);

      bool ruFullyReadout(sizes[iTower] == getElectronicsMap()->dccTowerConstituents(dccid, towerid).size());

      if (ruFullyReadout && (flag == EcalSrFlag::SRF_ZS1 || flag == EcalSrFlag::SRF_ZS2)) {
        meZSFullReadoutMap.fill(id);
        meZSFullReadout.fill(id);
      }

      if (sizes[iTower] == 0 && flag == EcalSrFlag::SRF_FULL) {
        meFRDroppedMap.fill(id);
        meFRDropped.fill(id);
      }
    }
  }

  void SelectiveReadoutTask::setFIRWeights_(std::vector<double> const& _normWeights) {
    if (_normWeights.size() < nFIRTaps)
      throw cms::Exception("InvalidConfiguration") << "weightsForZsFIR" << std::endl;

    bool notNormalized(false), notInt(false);
    for (std::vector<double>::const_iterator it(_normWeights.begin()); it != _normWeights.end(); ++it) {
      if (*it > 1.)
        notNormalized = true;
      if (int(*it) != *it)
        notInt = true;
    }
    if (notInt && notNormalized) {
      throw cms::Exception("InvalidConfiguration") << "weigtsForZsFIR paramater values are not valid: they "
                                                   << "must either be integer and uses the hardware representation "
                                                   << "of the weights or less or equal than 1 and used the normalized "
                                                   << "representation.";
    }

    ZSFIRWeights_.clear();
    ZSFIRWeights_.resize(_normWeights.size());

    if (notNormalized) {
      for (unsigned i(0); i < ZSFIRWeights_.size(); ++i)
        ZSFIRWeights_[i] = int(_normWeights[i]);
    } else {
      const int maxWeight(0xEFF);  //weights coded on 11+1 signed bits
      for (unsigned i(0); i < ZSFIRWeights_.size(); ++i) {
        ZSFIRWeights_[i] = lround(_normWeights[i] * (0x1 << 10));
        if (std::abs(ZSFIRWeights_[i]) > maxWeight)  //overflow
          ZSFIRWeights_[i] = ZSFIRWeights_[i] < 0 ? -maxWeight : maxWeight;
      }
    }
  }

  DEFINE_ECALDQM_WORKER(SelectiveReadoutTask);
}  // namespace ecaldqm
