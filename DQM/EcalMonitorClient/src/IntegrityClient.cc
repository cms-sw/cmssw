#include "../interface/IntegrityClient.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  IntegrityClient::IntegrityClient() :
    DQWorkerClient(),
    errFractionThreshold_(0.)
  {
    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
  }

  void
  IntegrityClient::setParams(edm::ParameterSet const& _params)
  {
    errFractionThreshold_ = _params.getUntrackedParameter<double>("errFractionThreshold");
  }

  void
  IntegrityClient::producePlots(ProcessType)
  {
    uint32_t mask(1 << EcalDQMStatusHelper::CH_ID_ERROR |
                  1 << EcalDQMStatusHelper::CH_GAIN_ZERO_ERROR |
                  1 << EcalDQMStatusHelper::CH_GAIN_SWITCH_ERROR |
                  1 << EcalDQMStatusHelper::TT_ID_ERROR |
                  1 << EcalDQMStatusHelper::TT_SIZE_ERROR);

    MESet& meQuality(MEs_.at("Quality"));
    MESet& meQualitySummary(MEs_.at("QualitySummary"));

    MESet const& sOccupancy(sources_.at("Occupancy"));
    MESet const& sGain(sources_.at("Gain"));
    MESet const& sChId(sources_.at("ChId"));
    MESet const& sGainSwitch(sources_.at("GainSwitch"));
    MESet const& sTowerId(sources_.at("TowerId"));
    MESet const& sBlockSize(sources_.at("BlockSize"));

    MESet::iterator qEnd(meQuality.end());
    MESet::const_iterator occItr(sOccupancy);
    for(MESet::iterator qItr(meQuality.beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      occItr = qItr;

      DetId id(qItr->getId());

      bool doMask(meQuality.maskMatches(id, mask, statusManager_));

      float entries(occItr->getBinContent());

      float gain(sGain.getBinContent(id));
      float chid(sChId.getBinContent(id));
      float gainswitch(sGainSwitch.getBinContent(id));

      float towerid(sTowerId.getBinContent(id));
      float blocksize(sBlockSize.getBinContent(id));

      if(entries + gain + chid + gainswitch + towerid + blocksize < 1.){
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        meQualitySummary.setBinContent(id, doMask ? kMUnknown : kUnknown);
        continue;
      }

      float chErr((gain + chid + gainswitch + towerid + blocksize) / (entries + gain + chid + gainswitch + towerid + blocksize));

      if(chErr > errFractionThreshold_){
        qItr->setBinContent(doMask ? kMBad : kBad);
        meQualitySummary.setBinContent(id, doMask ? kMBad : kBad);
      }
      else{
        qItr->setBinContent(doMask ? kMGood : kGood);
        meQualitySummary.setBinContent(id, doMask ? kMGood : kGood);
      }
    }
  }

  DEFINE_ECALDQM_WORKER(IntegrityClient);
}
