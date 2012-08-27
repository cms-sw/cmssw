#include "../interface/IntegrityClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

namespace ecaldqm {

  IntegrityClient::IntegrityClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "IntegrityClient"),
    errFractionThreshold_(_workerParams.getUntrackedParameter<double>("errFractionThreshold"))
  {
  }

  void
  IntegrityClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kQuality]->resetAll(-1.);
    MEs_[kQualitySummary]->resetAll(-1.);
  }

  void
  IntegrityClient::producePlots()
  {
    uint32_t mask(1 << EcalDQMStatusHelper::CH_ID_ERROR |
		  1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR |
		  1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR |
		  1 << EcalDQMStatusHelper::TT_ID_ERROR |
		  1 << EcalDQMStatusHelper::TT_SIZE_ERROR);

    MESet::iterator qEnd(MEs_[kQuality]->end());
    MESet::const_iterator occItr(sources_[kOccupancy]);
    for(MESet::iterator qItr(MEs_[kQuality]->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      occItr = qItr;

      DetId id(qItr->getId());

      float entries(occItr->getBinContent());

      float gain(sources_[kGain]->getBinContent(id));
      float chid(sources_[kChId]->getBinContent(id));
      float gainswitch(sources_[kGainSwitch]->getBinContent(id));

      float towerid(sources_[kTowerId]->getBinContent(id));
      float blocksize(sources_[kBlockSize]->getBinContent(id));

      if(entries + gain + chid + gainswitch + towerid + blocksize < 1.){
        qItr->setBinContent(maskQuality_(qItr, mask, 2));
        continue;
      }

      float chErr((gain + chid + gainswitch + towerid + blocksize) / (entries + gain + chid + gainswitch + towerid + blocksize));

      if(chErr > errFractionThreshold_)
        qItr->setBinContent(maskQuality_(qItr, mask, 0));
      else
        qItr->setBinContent(maskQuality_(qItr, mask, 1));
    }

    towerAverage_(kQualitySummary, kQuality, 0.5);
  }

  /*static*/
  void
  IntegrityClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["Quality"] = kQuality;
    _nameToIndex["QualitySummary"] = kQualitySummary;

    _nameToIndex["Occupancy"] = kOccupancy;
    _nameToIndex["Gain"] = kGain;
    _nameToIndex["ChId"] = kChId;
    _nameToIndex["GainSwitch"] = kGainSwitch;
    _nameToIndex["TowerId"] = kTowerId;
    _nameToIndex["BlockSize"] = kBlockSize;
  }

  DEFINE_ECALDQM_WORKER(IntegrityClient);
}
