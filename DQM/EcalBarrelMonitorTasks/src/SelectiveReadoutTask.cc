#include "../interface/SelectiveReadoutTask.h"

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

  SelectiveReadoutTask::SelectiveReadoutTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "SelectiveReadoutTask"),
    useCondDb_(_workerParams.getUntrackedParameter<bool>("useCondDb")),
    iFirstSample_(_workerParams.getUntrackedParameter<int>("DCCZS1stSample")),
    ZSFIRWeights_(nFIRTaps),
    suppressed_(),
    flags_(nRU, -1)
  {
    collectionMask_ = 
      (0x1 << kRun) |
      (0x1 << kSource) |
      (0x1 << kEcalRawData) |
      (0x1 << kEBSrFlag) |
      (0x1 << kEESrFlag) |
      (0x1 << kEBDigi) |
      (0x1 << kEEDigi);

    dependencies.push_back(Dependency(kEBDigi, kEcalRawData, kEBSrFlag));
    dependencies.push_back(Dependency(kEEDigi, kEcalRawData, kEESrFlag));

    if(!useCondDb_){
      std::vector<double> normWeights(_workerParams.getUntrackedParameter<std::vector<double> >("ZSFIRWeights"));
      setFIRWeights_(normWeights);
    }
  }

  void
  SelectiveReadoutTask::beginRun(const edm::Run &, const edm::EventSetup &_es)
  {
    using namespace std;

    if(useCondDb_){
      edm::ESHandle<EcalSRSettings> hSr;
      _es.get<EcalSRSettingsRcd>().get(hSr);

      vector<vector<float> > weights(hSr->dccNormalizedWeights_);
      if(weights.size() == 1){
	vector<double> normWeights;
	for(vector<float>::iterator it(weights[0].begin()); it != weights[0].end(); it++)
	  normWeights.push_back(*it);

	setFIRWeights_(normWeights);
      }
      else edm::LogWarning("EcalDQM") << "SelectiveReadoutTask: DCC weight set is not exactly 1.";
    }
  }

  void
  SelectiveReadoutTask::beginEvent(const edm::Event &, const edm::EventSetup &)
  {
    flags_.assign(nRU, -1);
    suppressed_.clear();
  }

  void
  SelectiveReadoutTask::runOnSource(const FEDRawDataCollection &_fedRaw)
  {
    float ebSize(0.), eeSize(0.);

    // DCC event size
    for(unsigned iFED(601); iFED <= 654; iFED++){
      float size(_fedRaw.FEDData(iFED).size() / 1024.);
      MEs_[kDCCSize]->fill(iFED - 600, size);
      if(iFED - 601 <= kEEmHigh || iFED - 601 >= kEEpLow) eeSize += size;
      else ebSize += size;
    }

    MEs_[kEventSize]->fill(unsigned(BinService::kEE) + 1, eeSize / 18.);
    MEs_[kEventSize]->fill(unsigned(BinService::kEB) + 1, ebSize / 36.);
  }

  void
  SelectiveReadoutTask::runOnRawData(const EcalRawDataCollection &_dcchs)
  {
    for(EcalRawDataCollection::const_iterator dcchItr(_dcchs.begin()); dcchItr != _dcchs.end(); ++dcchItr){
      std::vector<short> const& feStatus(dcchItr->getFEStatus());
      unsigned nFE(feStatus.size());
      for(unsigned iFE(0); iFE < nFE; ++iFE)
        if(feStatus[iFE] == Disabled) suppressed_.insert(std::make_pair(dcchItr->id(), iFE + 1));
    }
  }

  void
  SelectiveReadoutTask::runOnEBSrFlags(const EBSrFlagCollection &_srfs)
  {
    double nFR(0.);

    for(EBSrFlagCollection::const_iterator srfItr(_srfs.begin()); srfItr != _srfs.end(); ++srfItr)
      runOnSrFlag_(srfItr->id(), srfItr->value(), nFR);

    MEs_[kFullReadout]->fill(unsigned(BinService::kEB) + 1, nFR);
  }

  void
  SelectiveReadoutTask::runOnEESrFlags(const EESrFlagCollection &_srfs)
  {
    double nFR(0.);

    for(EESrFlagCollection::const_iterator srfItr(_srfs.begin()); srfItr != _srfs.end(); ++srfItr)
      runOnSrFlag_(srfItr->id(), srfItr->value(), nFR);

    MEs_[kFullReadout]->fill(unsigned(BinService::kEE) + 1, nFR);
  }

  void
  SelectiveReadoutTask::runOnSrFlag_(const DetId &_id, int _flag, double& _nFR)
  {
    MEs_[kFlagCounterMap]->fill(_id);

    unsigned iRU(-1);
    if(_id.subdetId() == EcalTriggerTower)
      iRU = EcalTrigTowerDetId(_id).hashedIndex();
    else
      iRU = EcalScDetId(_id).hashedIndex() + EcalTrigTowerDetId::kEBTotalTowers;
    flags_[iRU] = _flag;

    switch(_flag & ~EcalSrFlag::SRF_FORCED_MASK){
    case EcalSrFlag::SRF_FULL:
      MEs_[kFullReadoutMap]->fill(_id);
      _nFR += 1.;
      break;
    case EcalSrFlag::SRF_ZS1:
      MEs_[kZS1Map]->fill(_id);
      // fallthrough
    case EcalSrFlag::SRF_ZS2:
      MEs_[kZSMap]->fill(_id);
      break;
    default:
      break;
    }

    if(_flag & EcalSrFlag::SRF_FORCED_MASK)
      MEs_[kRUForcedMap]->fill(_id);
  }
  
  void
  SelectiveReadoutTask::runOnDigis(const EcalDigiCollection &_digis, Collections _collection)
  {
    std::vector<int> sizes(nRU, 0);

    int nHighInt(0), nLowInt(0);

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){

      DetId const& id(digiItr->id());

      unsigned iRU(-1);

      if(_collection == kEBDigi) iRU = EBDetId(id).tower().hashedIndex();
      else iRU = EEDetId(id).sc().hashedIndex() + EcalTrigTowerDetId::kEBTotalTowers;

      if(flags_[iRU] < 0) continue;

      sizes[iRU] += 1;

      // SR filter output calculation

      EcalDataFrame frame(*digiItr);

      int ZSFIRValue(0); // output

      bool gain12saturated(false);
      const int gain12(0x01);

      for(int iWeight(0); iWeight < nFIRTaps; ++iWeight){

	int iSample(iFirstSample_ + iWeight - 1);

	if(iSample >= 0 && iSample < frame.size()){
	  EcalMGPASample sample(frame[iSample]);
	  if(sample.gainId() != gain12){
	    gain12saturated = true;
	    break;
	  }
	  ZSFIRValue += sample.adc() * ZSFIRWeights_[iWeight];
	}else{
	  edm::LogWarning("EcalDQM") << "SelectiveReadoutTask: Not enough samples in data frame or 'ecalDccZs1stSample' module parameter is not valid";
	}

      }

      if(gain12saturated) ZSFIRValue = std::numeric_limits<int>::max();
      else ZSFIRValue /= (0x1 << 8); //discards the 8 LSBs

      //ZS passed if weighted sum above ZS threshold or if
      //one sample has a lower gain than gain 12 (that is gain 12 output
      //is saturated)

      bool highInterest((flags_[iRU] & ~EcalSrFlag::SRF_FORCED_MASK) == EcalSrFlag::SRF_FULL);

      if(highInterest){
	MEs_[kHighIntOutput]->fill(id, ZSFIRValue);
	nHighInt += 1;
      }
      else{
	MEs_[kLowIntOutput]->fill(id, ZSFIRValue);
	nLowInt += 1;
      }
    }

    unsigned iSubdet(_collection == kEBDigi ? BinService::kEB : BinService::kEE);
    float denom(_collection == kEBDigi ? 36. : 18.);

    float highIntPayload(nHighInt * bytesPerCrystal / 1024. / denom);
    MEs_[kHighIntPayload]->fill(iSubdet + 1, highIntPayload);

    float lowIntPayload(nLowInt * bytesPerCrystal / 1024. / denom);
    MEs_[kLowIntPayload]->fill(iSubdet + 1, lowIntPayload);

    float nZSFullReadout(0.);
    float nFRDropped(0.);

    for(unsigned iRU(0); iRU < nRU; ++iRU){
      DetId id;
      if(iRU < EcalTrigTowerDetId::kEBTotalTowers) id = EcalTrigTowerDetId::detIdFromDenseIndex(iRU);
      else id = EcalScDetId::unhashIndex(iRU - EcalTrigTowerDetId::kEBTotalTowers);

      double towerSize(sizes[iRU] * bytesPerCrystal);

      MEs_[kTowerSize]->fill(id, towerSize);

      if(flags_[iRU] < 0) continue;

      int dccid(dccId(id));
      int towerid(towerId(id));

      if(suppressed_.find(std::make_pair(dccid, towerid)) != suppressed_.end()) continue;

      int flag(flags_[iRU] & ~EcalSrFlag::SRF_FORCED_MASK);

      bool ruFullyReadout(unsigned(sizes[iRU]) == getElectronicsMap()->dccTowerConstituents(dccid, towerid).size());

      if(ruFullyReadout && (flag == EcalSrFlag::SRF_ZS1 || flag == EcalSrFlag::SRF_ZS2)){
	MEs_[kZSFullReadoutMap]->fill(id);
	nZSFullReadout += 1.;
      }

      if(sizes[iRU] == 0 && flag == EcalSrFlag::SRF_FULL){
        MEs_[kFRDroppedMap]->fill(id);
        nFRDropped += 1.;
      }
    }

    MEs_[kZSFullReadout]->fill(iSubdet + 1, nZSFullReadout);
    MEs_[kFRDropped]->fill(iSubdet + 1, nFRDropped);
  }

  void
  SelectiveReadoutTask::setFIRWeights_(const std::vector<double> &_normWeights)
  {
    if(_normWeights.size() < nFIRTaps)
      throw cms::Exception("InvalidConfiguration") << "weightsForZsFIR" << std::endl;

    bool notNormalized(false), notInt(false);
    for(std::vector<double>::const_iterator it(_normWeights.begin()); it != _normWeights.end(); ++it){
      if(*it > 1.) notNormalized = true;
      if(int(*it) != *it) notInt = true;
    }
    if(notInt && notNormalized){
      throw cms::Exception("InvalidConfiguration")
	<< "weigtsForZsFIR paramater values are not valid: they "
	<< "must either be integer and uses the hardware representation "
	<< "of the weights or less or equal than 1 and used the normalized "
	<< "representation.";
    }

    ZSFIRWeights_.clear();
    ZSFIRWeights_.resize(_normWeights.size());

    if(notNormalized){
      for(unsigned i(0); i< ZSFIRWeights_.size(); ++i)
	ZSFIRWeights_[i] = int(_normWeights[i]);
    }
    else{
      const int maxWeight(0xEFF); //weights coded on 11+1 signed bits
      for(unsigned i(0); i < ZSFIRWeights_.size(); ++i){
	ZSFIRWeights_[i] = lround(_normWeights[i] * (0x1 << 10));
	if(std::abs(ZSFIRWeights_[i]) > maxWeight) //overflow
	  ZSFIRWeights_[i] = ZSFIRWeights_[i] < 0 ? -maxWeight : maxWeight;
      }
    }
  }

  /*static*/
  void
  SelectiveReadoutTask::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["TowerSize"] = kTowerSize;
    _nameToIndex["DCCSize"] = kDCCSize;
    _nameToIndex["EventSize"] = kEventSize;
    _nameToIndex["FlagCounterMap"] = kFlagCounterMap;
    _nameToIndex["RUForcedMap"] = kRUForcedMap;
    _nameToIndex["FullReadout"] = kFullReadout;
    _nameToIndex["FullReadoutMap"] = kFullReadoutMap;
    _nameToIndex["ZS1Map"] = kZS1Map;
    _nameToIndex["ZSMap"] = kZSMap;
    _nameToIndex["ZSFullReadout"] = kZSFullReadout;
    _nameToIndex["ZSFullReadoutMap"] = kZSFullReadoutMap;
    _nameToIndex["FRDropped"] = kFRDropped;
    _nameToIndex["FRDroppedMap"] = kFRDroppedMap;
    _nameToIndex["HighIntPayload"] = kHighIntPayload;
    _nameToIndex["LowIntPayload"] = kLowIntPayload;
    _nameToIndex["HighIntOutput"] = kHighIntOutput;
    _nameToIndex["LowIntOutput"] = kLowIntOutput;
  }

  DEFINE_ECALDQM_WORKER(SelectiveReadoutTask);
}


