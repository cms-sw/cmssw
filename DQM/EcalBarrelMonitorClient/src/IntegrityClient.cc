#include "../interface/IntegrityClient.h"
#include "../interface/EcalDQMClientUtils.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

namespace ecaldqm {

  IntegrityClient::IntegrityClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "IntegrityClient"),
    errFractionThreshold_(_workerParams.getUntrackedParameter<double>("errFractionThreshold"))
  {
    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
  }

  void
  IntegrityClient::producePlots()
  {
    uint32_t mask(1 << EcalDQMStatusHelper::CH_ID_ERROR |
		  1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR |
		  1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR |
		  1 << EcalDQMStatusHelper::TT_ID_ERROR |
		  1 << EcalDQMStatusHelper::TT_SIZE_ERROR);

    MESet* meQuality(MEs_["Quality"]);
    MESet* meQualitySummary(MEs_["QualitySummary"]);

    MESet const* sOccupancy(sources_["Occupancy"]);
    MESet const* sGain(sources_["Gain"]);
    MESet const* sChId(sources_["ChId"]);
    MESet const* sGainSwitch(sources_["GainSwitch"]);
    MESet const* sTowerId(sources_["TowerId"]);
    MESet const* sBlockSize(sources_["BlockSize"]);

    MESet::iterator qEnd(meQuality->end());
    MESet::const_iterator occItr(sOccupancy);
    for(MESet::iterator qItr(meQuality->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      occItr = qItr;

      DetId id(qItr->getId());

      bool doMask(applyMask(meQuality->getBinType(), id, mask));

      float entries(occItr->getBinContent());

      float gain(sGain->getBinContent(id));
      float chid(sChId->getBinContent(id));
      float gainswitch(sGainSwitch->getBinContent(id));

      float towerid(sTowerId->getBinContent(id));
      float blocksize(sBlockSize->getBinContent(id));

      if(entries + gain + chid + gainswitch + towerid + blocksize < 1.){
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        meQualitySummary->setBinContent(id, doMask ? kMUnknown : kUnknown);
        continue;
      }

      float chErr((gain + chid + gainswitch + towerid + blocksize) / (entries + gain + chid + gainswitch + towerid + blocksize));

      if(chErr > errFractionThreshold_){
        qItr->setBinContent(doMask ? kMBad : kBad);
        meQualitySummary->setBinContent(id, doMask ? kMBad : kBad);
      }
      else{
        qItr->setBinContent(doMask ? kMGood : kGood);
        meQualitySummary->setBinContent(id, doMask ? kMGood : kGood);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(IntegrityClient);
}
