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
    collectionMask_[kRun] = true;
    collectionMask_[kSource] = true;
    collectionMask_[kEcalRawData] = true;
    collectionMask_[kEBSrFlag] = true;
    collectionMask_[kEESrFlag] = true;
    collectionMask_[kEBDigi] = true;
    collectionMask_[kEEDigi] = true;

    if(!useCondDb_){
      std::vector<double> normWeights(_workerParams.getUntrackedParameter<std::vector<double> >("ZSFIRWeights"));
      setFIRWeights_(normWeights);
    }
  }

  void
  SelectiveReadoutTask::setDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kEBDigi, kEcalRawData, kEBSrFlag));
    _dependencies.push_back(Dependency(kEEDigi, kEcalRawData, kEESrFlag));
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
    MESet* meDCCSize(MEs_["DCCSize"]);
    MESet* meDCCSizeProf(MEs_["DCCSizeProf"]);
    MESet* meEventSize(MEs_["EventSize"]);

    float ebSize(0.), eemSize(0.), eepSize(0.);

    // DCC event size
    for(unsigned iFED(601); iFED <= 654; iFED++){
      float size(_fedRaw.FEDData(iFED).size() / 1024.);
      meDCCSize->fill(iFED - 600, size);
      meDCCSizeProf->fill(iFED - 600, size);
      if(iFED - 601 <= kEEmHigh) eemSize += size;
      else if(iFED - 601 >= kEEpLow) eepSize += size;
      else ebSize += size;
    }

    meEventSize->fill(unsigned(BinService::kEEm) + 1, eemSize / 9.);
    meEventSize->fill(unsigned(BinService::kEEp) + 1, eepSize / 9.);
    meEventSize->fill(unsigned(BinService::kEB) + 1, ebSize / 36.);
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
    MESet* meFullReadout(MEs_["FullReadout"]);
    MESet* maps[] = {MEs_["FlagCounterMap"], MEs_["FullReadoutMap"], MEs_["ZS1Map"], MEs_["ZSMap"], MEs_["RUForcedMap"]};

    double nFR(0.);

    for(EBSrFlagCollection::const_iterator srfItr(_srfs.begin()); srfItr != _srfs.end(); ++srfItr)
      runOnSrFlag_(srfItr->id(), srfItr->value(), nFR, maps);

    meFullReadout->fill(unsigned(BinService::kEB) + 1, nFR);
  }

  void
  SelectiveReadoutTask::runOnEESrFlags(const EESrFlagCollection &_srfs)
  {
    MESet* meFullReadout(MEs_["FullReadout"]);
    MESet* maps[] = {MEs_["FlagCounterMap"], MEs_["FullReadoutMap"], MEs_["ZS1Map"], MEs_["ZSMap"], MEs_["RUForcedMap"]};

    double nFR(0.);

    for(EESrFlagCollection::const_iterator srfItr(_srfs.begin()); srfItr != _srfs.end(); ++srfItr)
      runOnSrFlag_(srfItr->id(), srfItr->value(), nFR, maps);

    meFullReadout->fill(unsigned(BinService::kEE) + 1, nFR);
  }

  void
  SelectiveReadoutTask::runOnSrFlag_(const DetId &_id, int _flag, double& _nFR, MESet** _maps)
  {
    MESet* meFlagCounterMap(_maps[0]);
    MESet* meFullReadoutMap(_maps[1]);
    MESet* meZS1Map(_maps[2]);
    MESet* meZSMap(_maps[3]);
    MESet* meRUForcedMap(_maps[4]);

    meFlagCounterMap->fill(_id);

    unsigned iRU(-1);
    if(_id.subdetId() == EcalTriggerTower)
      iRU = EcalTrigTowerDetId(_id).hashedIndex();
    else
      iRU = EcalScDetId(_id).hashedIndex() + EcalTrigTowerDetId::kEBTotalTowers;
    flags_[iRU] = _flag;

    switch(_flag & ~EcalSrFlag::SRF_FORCED_MASK){
    case EcalSrFlag::SRF_FULL:
      meFullReadoutMap->fill(_id);
      _nFR += 1.;
      break;
    case EcalSrFlag::SRF_ZS1:
      meZS1Map->fill(_id);
      // fallthrough
    case EcalSrFlag::SRF_ZS2:
      meZSMap->fill(_id);
      break;
    default:
      break;
    }

    if(_flag & EcalSrFlag::SRF_FORCED_MASK)
      meRUForcedMap->fill(_id);
  }
  
  void
  SelectiveReadoutTask::runOnDigis(const EcalDigiCollection &_digis, Collections _collection)
  {
    MESet* meHighIntOutput(MEs_["HighIntOutput"]);
    MESet* meLowIntOutput(MEs_["LowIntOutput"]);
    MESet* meHighIntPayload(MEs_["HighIntPayload"]);
    MESet* meLowIntPayload(MEs_["LowIntPayload"]);
    MESet* meTowerSize(MEs_["TowerSize"]);
    MESet* meZSFullReadoutMap(MEs_["ZSFullReadoutMap"]);
    MESet* meFRDroppedMap(MEs_["FRDroppedMap"]);
    MESet* meZSFullReadout(MEs_["ZSFullReadout"]);
    MESet* meFRDropped(MEs_["FRDropped"]);

    bool isEB(_collection == kEBDigi);

    unsigned const nTower(isEB ? unsigned(EcalTrigTowerDetId::kEBTotalTowers) : unsigned(EcalScDetId::kSizeForDenseIndexing));

    std::vector<unsigned> sizes(nTower, 0);

    int nHighInt[] = {0, 0};
    int nLowInt[] = {0, 0};

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){

      DetId const& id(digiItr->id());

      unsigned iTower(-1);
      unsigned iRU(-1);

      if(isEB){
        iTower = EBDetId(id).tower().hashedIndex();
        iRU = iTower;
      }
      else{
        iTower = EEDetId(id).sc().hashedIndex();
        iRU = iTower + EcalTrigTowerDetId::kEBTotalTowers;
      }

      if(flags_[iRU] < 0) continue;

      sizes[iTower] += 1;

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
	meHighIntOutput->fill(id, ZSFIRValue);
        if(isEB || dccId(id) - 1 <= kEEmHigh)
          nHighInt[0] += 1;
        else
          nHighInt[1] += 1;
      }
      else{
	meLowIntOutput->fill(id, ZSFIRValue);
        if(isEB || dccId(id) - 1 <= kEEmHigh)
          nLowInt[0] += 1;
        else
          nLowInt[1] += 1;
      }
    }

    float denom(_collection == kEBDigi ? 36. : 9.);

    if(isEB){
      meHighIntPayload->fill(unsigned(BinService::kEB + 1), nHighInt[0] * bytesPerCrystal / 1024. / 36.);
      meLowIntPayload->fill(unsigned(BinService::kEB + 1), nLowInt[0] * bytesPerCrystal / 1024. / 36.);
    }
    else{
      meHighIntPayload->fill(unsigned(BinService::kEEm + 1), nHighInt[0] * bytesPerCrystal / 1024. / 9.);
      meHighIntPayload->fill(unsigned(BinService::kEEp + 1), nHighInt[1] * bytesPerCrystal / 1024. / 9.);
      meLowIntPayload->fill(unsigned(BinService::kEEm + 1), nLowInt[0] * bytesPerCrystal / 1024. / 9.);
      meLowIntPayload->fill(unsigned(BinService::kEEp + 1), nLowInt[1] * bytesPerCrystal / 1024. / 9.);
    }

    unsigned iRU(isEB ? 0 : EcalTrigTowerDetId::kEBTotalTowers);
    for(unsigned iTower(0); iTower < nTower; ++iTower, ++iRU){
      DetId id;
      if(isEB) id = EcalTrigTowerDetId::detIdFromDenseIndex(iTower);
      else id = EcalScDetId::unhashIndex(iTower);

      double towerSize(sizes[iTower] * bytesPerCrystal);

      meTowerSize->fill(id, towerSize);

      if(flags_[iRU] < 0) continue;

      int dccid(dccId(id));
      int towerid(towerId(id));

      if(suppressed_.find(std::make_pair(dccid, towerid)) != suppressed_.end()) continue;

      int flag(flags_[iRU] & ~EcalSrFlag::SRF_FORCED_MASK);

      bool ruFullyReadout(sizes[iTower] == getElectronicsMap()->dccTowerConstituents(dccid, towerid).size());

      if(ruFullyReadout && (flag == EcalSrFlag::SRF_ZS1 || flag == EcalSrFlag::SRF_ZS2)){
	meZSFullReadoutMap->fill(id);
        meZSFullReadout->fill(id);
      }

      if(sizes[iTower] == 0 && flag == EcalSrFlag::SRF_FULL){
        meFRDroppedMap->fill(id);
        meFRDropped->fill(id);
      }
    }
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

  DEFINE_ECALDQM_WORKER(SelectiveReadoutTask);
}


