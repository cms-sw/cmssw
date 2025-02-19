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

  SelectiveReadoutTask::SelectiveReadoutTask(const edm::ParameterSet &_params, const edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "SelectiveReadoutTask"),
    useCondDb_(false),
    iFirstSample_(0),
    channelStatus_(0),
    ebSRFs_(0),
    eeSRFs_(0),
    frFlaggedTowers_(),
    zsFlaggedTowers_()
  {
    collectionMask_ = 
      (0x1 << kRun) |
      (0x1 << kSource) |
      (0x1 << kEcalRawData) |
      (0x1 << kEBSrFlag) |
      (0x1 << kEESrFlag) |
      (0x1 << kEBDigi) |
      (0x1 << kEEDigi);

    dependencies_.push_back(std::pair<Collections, Collections>(kEBDigi, kEcalRawData));
    dependencies_.push_back(std::pair<Collections, Collections>(kEEDigi, kEcalRawData));

    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));

    useCondDb_ = taskParams.getUntrackedParameter<bool>("useCondDb");
    iFirstSample_ = taskParams.getUntrackedParameter<int>("DCCZS1stSample");

    std::vector<double> normWeights(taskParams.getUntrackedParameter<std::vector<double> >("ZSFIRWeights", std::vector<double>(0)));
    if(normWeights.size()) setFIRWeights_(normWeights);
  }

  SelectiveReadoutTask::~SelectiveReadoutTask()
  {
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

    edm::ESHandle<EcalChannelStatus> chSHndl;
    _es.get<EcalChannelStatusRcd>().get(chSHndl);
    channelStatus_ = chSHndl.product();
    if(!channelStatus_)
      throw cms::Exception("EventSetup") << "EcalChannelStatusRcd";
  }

  void
  SelectiveReadoutTask::beginEvent(const edm::Event &, const edm::EventSetup &)
  {
    for(int iDCC(0); iDCC < 54; iDCC++) feStatus_[iDCC].clear();
    frFlaggedTowers_.clear();
    zsFlaggedTowers_.clear();
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
      const std::vector<short> &feStatus(dcchItr->getFEStatus());
      feStatus_[dcchItr->id() - 1].assign(feStatus.begin(), feStatus.end());
    }
  }

  void
  SelectiveReadoutTask::runOnEBSrFlags(const EBSrFlagCollection &_srfs)
  {
    float nFR(0.);

    ebSRFs_ = &_srfs;

    for(EBSrFlagCollection::const_iterator srfItr(_srfs.begin()); srfItr != _srfs.end(); ++srfItr)
      runOnSrFlag_(srfItr->id(), srfItr->value(), nFR);

    MEs_[kFullReadout]->fill(unsigned(BinService::kEB) + 1, nFR);
  }

  void
  SelectiveReadoutTask::runOnEESrFlags(const EESrFlagCollection &_srfs)
  {
    float nFR(0.);

    eeSRFs_ = &_srfs;

    for(EESrFlagCollection::const_iterator srfItr(_srfs.begin()); srfItr != _srfs.end(); ++srfItr)
      runOnSrFlag_(srfItr->id(), srfItr->value(), nFR);

    MEs_[kFullReadout]->fill(unsigned(BinService::kEE) + 1, nFR);
  }

  void
  SelectiveReadoutTask::runOnSrFlag_(const DetId &_id, int _flag, float& nFR)
  {
    uint32_t rawId(_id.rawId());
    int dccid(dccId(_id));
    int towerid(towerId(_id));

    MEs_[kFlagCounterMap]->fill(_id);

    short status(feStatus_[dccid - 1].size() ? feStatus_[dccid - 1][towerid - 1] : 0); // check: towerid == feId??
 
    switch(_flag & ~EcalSrFlag::SRF_FORCED_MASK){
    case EcalSrFlag::SRF_FULL:
      MEs_[kFullReadoutMap]->fill(_id);
      nFR += 1.;
      if(status != Disabled) frFlaggedTowers_.insert(rawId); // will be used in Digi loop
      break;
    case EcalSrFlag::SRF_ZS1:
      MEs_[kZS1Map]->fill(_id);
      // fallthrough
    case EcalSrFlag::SRF_ZS2:
      MEs_[kZSMap]->fill(_id);
      if(status != Disabled) zsFlaggedTowers_.insert(rawId);
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
    using namespace std;

    map<uint32_t, pair<int, int> > flagAndSizeMap;
    map<uint32_t, pair<int, int> >::iterator fasItr;

    int nHighInt(0), nLowInt(0); // one or two entries will be empty

    for(EcalDigiCollection::const_iterator digiItr(_digis.begin()); digiItr != _digis.end(); ++digiItr){

      DetId id(digiItr->id());

      pair<int, int> *flagAndSize(0);

      if(_collection == kEBDigi){
	EcalTrigTowerDetId ttid(EBDetId(id).tower());
	uint32_t rawId(ttid.rawId());

	fasItr = flagAndSizeMap.find(rawId);

	if(fasItr == flagAndSizeMap.end()){
	  flagAndSize = &(flagAndSizeMap[rawId]);

	  EBSrFlagCollection::const_iterator srItr(ebSRFs_->find(ttid));
	  if(srItr != ebSRFs_->end()) flagAndSize->first = srItr->value();
	  else flagAndSize->first = -1;
	}else{
	  flagAndSize = &(fasItr->second);
	}
      }else{
	EcalScDetId scid(EEDetId(id).sc());
	uint32_t rawId(scid.rawId());

	fasItr = flagAndSizeMap.find(rawId);

	if(fasItr == flagAndSizeMap.end()){
	  flagAndSize = &(flagAndSizeMap[rawId]);

	  EESrFlagCollection::const_iterator srItr(eeSRFs_->find(scid));
	  if(srItr != eeSRFs_->end()) flagAndSize->first = srItr->value();
	  else flagAndSize->first = -1;
	}else{
	  flagAndSize = &(fasItr->second);
	}
      }

      if(flagAndSize->first < 0) continue;

      flagAndSize->second += 1;

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

      bool highInterest((flagAndSize->first & ~EcalSrFlag::SRF_FORCED_MASK) == EcalSrFlag::SRF_FULL);

      if(highInterest){
	MEs_[kHighIntOutput]->fill(id, ZSFIRValue);
	nHighInt += 1;
      }else{
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

    // Check for "ZS-flagged but readout" and "FR-flagged but dropped" towers

    float nZSFullReadout(0.);
    for(unsigned iTower(0); iTower < EcalTrigTowerDetId::kEBTotalTowers + EcalScDetId::kSizeForDenseIndexing; iTower++){
      DetId id;
      if(iTower < EcalTrigTowerDetId::kEBTotalTowers) id = EcalTrigTowerDetId::detIdFromDenseIndex(iTower);
      else id = EcalScDetId::unhashIndex(iTower - EcalTrigTowerDetId::kEBTotalTowers);

      fasItr = flagAndSizeMap.find(id.rawId());

      float towerSize(0.);
      if(fasItr != flagAndSizeMap.end()) towerSize = fasItr->second.second * bytesPerCrystal;

      MEs_[kTowerSize]->fill(id, towerSize);

      if(fasItr == flagAndSizeMap.end() || fasItr->second.first < 0) continue; // not read out || no flag set

      bool ruFullyReadout(unsigned(fasItr->second.second) == getElectronicsMap()->dccTowerConstituents(dccId(id), towerId(id)).size());

      if(ruFullyReadout && zsFlaggedTowers_.find(id.rawId()) != zsFlaggedTowers_.end()){
	MEs_[kZSFullReadoutMap]->fill(id);
	nZSFullReadout += 1.;
      }

      // we will later use the list of FR flagged towers that do not have data
      // if the tower is in flagAndSizeMap then there is data; remove it from the list
      if(frFlaggedTowers_.find(id.rawId()) != frFlaggedTowers_.end()) frFlaggedTowers_.erase(id.rawId());
    }

    MEs_[kZSFullReadout]->fill(iSubdet + 1, nZSFullReadout);

    float nFRDropped(0.);

    for(set<uint32_t>::iterator frItr(frFlaggedTowers_.begin()); frItr != frFlaggedTowers_.end(); ++frItr){
      DetId id(*frItr);

      MEs_[kFRDroppedMap]->fill(id);
      nFRDropped += 1.;
    }

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
      if((int)(*it) != *it) notInt = true;
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
	ZSFIRWeights_[i] = (int)_normWeights[i];
    }else{
      const unsigned maxWeight(0xEFF); //weights coded on 11+1 signed bits
      for(unsigned i(0); i < ZSFIRWeights_.size(); ++i){
	ZSFIRWeights_[i] = lround(_normWeights[i] * (1<<10));
	if(abs(ZSFIRWeights_[i]) > (int)maxWeight) //overflow
	  ZSFIRWeights_[i] = ZSFIRWeights_[i] < 0 ? -maxWeight : maxWeight;
      }
    }
  }

  /*static*/
  void
  SelectiveReadoutTask::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs axis;

    axis.low = 0.;
    axis.high = 50.;
    _data[kTowerSize] = MEData("TowerSize", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &axis);

    axis.title = "event size (kB)";
    axis.nbins = 78; // 10 zero-bins + 68
    axis.edges = new double[79];
    float fullTTSize(0.608);
    for(int i(0); i <= 10; i++) axis.edges[i] = fullTTSize / 10. * i;
    for(int i(11); i < 79; i++) axis.edges[i] = fullTTSize * (i - 10);
    _data[kDCCSize] = MEData("DCCSize", BinService::kEcal2P, BinService::kDCC, MonitorElement::DQM_KIND_TH2F, 0, &axis);
    delete [] axis.edges;
    axis.edges = 0;

    axis.nbins = 100;
    axis.low = 0.;
    axis.high = 3.;
    axis.title = "event size (kB)";
    _data[kEventSize] = MEData("EventSize", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kFlagCounterMap] = MEData("FlagCounterMap", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kRUForcedMap] = MEData("RUForcedMap", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);

    axis.nbins = 100;
    axis.low = 0.;
    axis.high = 200.;
    axis.title = "number of towers";
    _data[kFullReadout] = MEData("FullReadout", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kFullReadoutMap] = MEData("FullReadoutMap", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kZS1Map] = MEData("ZS1Map", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kZSMap] = MEData("ZSMap", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);

    axis.nbins = 20;
    axis.low = 0.;
    axis.high = 20.;
    axis.title = "number of towers";
    _data[kZSFullReadout] = MEData("ZSFullReadout", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kZSFullReadoutMap] = MEData("ZSFullReadoutMap", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kFRDropped] = MEData("FRDropped", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kFRDroppedMap] = MEData("FRDroppedMap", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);

    axis.nbins = 100;
    axis.low = 0.;
    axis.high = 3.;
    axis.title = "event size (kB)";
    _data[kHighIntPayload] = MEData("HighIntPayload", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kLowIntPayload] = MEData("LowIntPayload", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);

    axis.nbins = 100;
    axis.low = -60.;
    axis.high = 60.;
    axis.title = "ADC counts*4";
    _data[kHighIntOutput] = MEData("HighIntOutput", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
    _data[kLowIntOutput] = MEData("LowIntOutput", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &axis);
  }

  DEFINE_ECALDQM_WORKER(SelectiveReadoutTask);
}


