#include "../interface/PresampleClient.h"
#include "../interface/EcalDQMClientUtils.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include <cmath>

namespace ecaldqm {

  PresampleClient::PresampleClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "PresampleClient"),
    minChannelEntries_(_workerParams.getUntrackedParameter<int>("minChannelEntries")),
    expectedMean_(_workerParams.getUntrackedParameter<double>("expectedMean")),
    toleranceMean_(_workerParams.getUntrackedParameter<double>("toleranceMean")),
    toleranceRMS_(_workerParams.getUntrackedParameter<double>("toleranceRMS")),
    toleranceRMSFwd_(_workerParams.getUntrackedParameter<double>("toleranceRMSFwd"))
  {
    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
  }

  void
  PresampleClient::producePlots()
  {
    MESet* meQualitySummary(MEs_["QualitySummary"]);
    MESet* meQuality(MEs_["Quality"]);
    MESet* meMean(MEs_["Mean"]);
    MESet* meRMS(MEs_["RMS"]);
    MESet* meRMSMap(MEs_["RMSMap"]);
    MESet* meRMSMapAll(MEs_["RMSMapAll"]);
    MESet* meTrendMean(online ? MEs_["TrendMean"] : 0);
    MESet* meTrendRMS(online ? MEs_["TrendRMS"] : 0);

    MESet const* sPedestal(sources_["Pedestal"]);

    meMean->reset();
    meRMS->reset();
    meRMSMapAll->reset(-1.);

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
		  1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR);

    MESet::iterator qEnd(meQuality->end());

    MESet::const_iterator pItr(sPedestal);
    double maxEB(0.), minEB(0.), maxEE(0.), minEE(0.);
    double rmsMaxEB(0.), rmsMaxEE(0.);
    for(MESet::iterator qItr(meQuality->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      pItr = qItr;

      DetId id(qItr->getId());

      bool doMask(applyMask(meQuality->getBinType(), id, mask));

      double rmsThresh(toleranceRMS_);

      if(isForward(id)) rmsThresh = toleranceRMSFwd_;

      double entries(pItr->getBinEntries());

      if(entries < minChannelEntries_){
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        meQualitySummary->setBinContent(id, doMask ? kMUnknown : kUnknown);
        meRMSMap->setBinContent(id, -1.);
        continue;
      }

      double mean(pItr->getBinContent());
      double rms(pItr->getBinError() * std::sqrt(entries));

      unsigned dccid(dccId(id));

      meMean->fill(dccid, mean);
      meRMS->fill(dccid, rms);
      meRMSMap->setBinContent(id, rms);

      if(std::abs(mean - expectedMean_) > toleranceMean_ || rms > rmsThresh){
        qItr->setBinContent(doMask ? kMBad : kBad);
        meQualitySummary->setBinContent(id, doMask ? kMBad : kBad);
      }
      else{
        qItr->setBinContent(doMask ? kMGood : kGood);
        meQualitySummary->setBinContent(id, doMask ? kMGood : kGood);
      }

      if(id.subdetId() == EcalBarrel){
        if(mean > maxEB) maxEB = mean;
        if(mean < minEB) minEB = mean;
        if(rms > rmsMaxEB) rmsMaxEB = rms;
      }
      else{
        if(mean > maxEE) maxEE = mean;
        if(mean < minEE) minEE = mean;
        if(rms > rmsMaxEE) rmsMaxEE = rms;
      }
    }

    towerAverage_(meRMSMapAll, meRMSMap, -1.);

    if(online){
      meTrendMean->fill(unsigned(BinService::kEB + 1), double(iLumi), maxEB - minEB);
      meTrendMean->fill(unsigned(BinService::kEE + 1), double(iLumi), maxEE - minEE);
      meTrendRMS->fill(unsigned(BinService::kEB + 1), double(iLumi), rmsMaxEB);
      meTrendRMS->fill(unsigned(BinService::kEE + 1), double(iLumi), rmsMaxEE);
    }
  }

  DEFINE_ECALDQM_WORKER(PresampleClient);
}


