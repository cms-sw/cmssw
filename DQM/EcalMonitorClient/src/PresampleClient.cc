#include "../interface/PresampleClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>

namespace ecaldqm
{
  PresampleClient::PresampleClient() :
    DQWorkerClient(),
    minChannelEntries_(0),
    expectedMean_(0.),
    toleranceMean_(0.),
    toleranceRMS_(0.),
    toleranceRMSFwd_(0.)
  {
    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
  }

  void
  PresampleClient::setParams(edm::ParameterSet const& _params)
  {
    minChannelEntries_ = _params.getUntrackedParameter<int>("minChannelEntries");
    expectedMean_ = _params.getUntrackedParameter<double>("expectedMean");
    toleranceMean_ = _params.getUntrackedParameter<double>("toleranceMean");
    toleranceRMS_ = _params.getUntrackedParameter<double>("toleranceRMS");
    toleranceRMSFwd_ = _params.getUntrackedParameter<double>("toleranceRMSFwd");
  }

  void
  PresampleClient::producePlots(ProcessType)
  {
    MESet& meQualitySummary(MEs_.at("QualitySummary"));
    MESet& meQuality(MEs_.at("Quality"));
    MESet& meErrorsSummary(MEs_.at("ErrorsSummary"));
    MESet& meMean(MEs_.at("Mean"));
    MESet& meRMS(MEs_.at("RMS"));
    MESet& meRMSMap(MEs_.at("RMSMap"));
    MESet& meRMSMapAll(MEs_.at("RMSMapAll"));
    MESet& meRMSMapAllByLumi(MEs_.at("RMSMapAllByLumi"));

    MESet const& sPedestal(sources_.at("Pedestal"));
    MESet const& sPedestalByLS(sources_.at("PedestalByLS"));
    MESet const& sChStatus(sources_.at("ChStatus"));

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_MEAN_ERROR |
		  1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR);

    MESet::iterator qEnd(meQuality.end());

    MESet::const_iterator pItr(sPedestal);
    MESet::const_iterator pLSItr(sPedestalByLS);
    double maxEB(0.), minEB(0.), maxEE(0.), minEE(0.);
    double rmsMaxEB(0.), rmsMaxEE(0.);
    for(MESet::iterator qItr(meQuality.beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      pItr = qItr;
      pLSItr = qItr;

      DetId id(qItr->getId());

      bool doMask(meQuality.maskMatches(id, mask, statusManager_));

      double rmsThresh(toleranceRMS_);

      if(isForward(id)) rmsThresh = toleranceRMSFwd_;

      double entries(pItr->getBinEntries());
      double entriesLS(pLSItr->getBinEntries());

      if(entries < minChannelEntries_){
        qItr->setBinContent(doMask ? kMUnknown : kUnknown);
        meQualitySummary.setBinContent(id, doMask ? kMUnknown : kUnknown);
        meRMSMap.setBinContent(id, -1.);
        continue;
      }

      double mean(pItr->getBinContent());
      double meanLS(pLSItr->getBinContent());
      double rms(pItr->getBinError() * std::sqrt(entries));
      double rmsLS(pLSItr->getBinError() * std::sqrt(entriesLS));

      int dccid(dccId(id));

      meMean.fill(dccid, mean);
      meRMS.fill(dccid, rms);
      meRMSMap.setBinContent(id, rms);
      meRMSMapAllByLumi.setBinContent(id, rmsLS);

      if(std::abs(mean - expectedMean_) > toleranceMean_ || rms > rmsThresh){
        qItr->setBinContent(doMask ? kMBad : kBad);
        meQualitySummary.setBinContent(id, doMask ? kMBad : kBad);
        if(!doMask) meErrorsSummary.fill(id);
      }
      else{
        qItr->setBinContent(doMask ? kMGood : kGood);
        meQualitySummary.setBinContent(id, doMask ? kMGood : kGood);
      }

      // Fill Presample Trend plots:
      // Use PedestalByLS which only contains digis from "current" LS
      float  chStatus( sChStatus.getBinContent(id) );
      if ( entriesLS < minChannelEntries_ ) continue;
      if ( chStatus != EcalChannelStatusCode::kOk ) continue; // exclude problematic channels

      // Get max/min
      // Min is effectively just 0
      if( id.subdetId() == EcalBarrel ){
        if( meanLS > maxEB ) maxEB = meanLS;
        if( meanLS < minEB ) minEB = meanLS;
        if( rmsLS > rmsMaxEB ) rmsMaxEB = rmsLS;
      }
      else {
        if( meanLS > maxEE ) maxEE = meanLS;
        if( meanLS < minEE ) minEE = meanLS;
        if( rmsLS > rmsMaxEE ) rmsMaxEE = rmsLS;
      }

    } // qItr

    towerAverage_(meRMSMapAll, meRMSMap, -1.);

    MESet& meTrendMean(MEs_.at("TrendMean"));
    MESet& meTrendRMS(MEs_.at("TrendRMS"));
    meTrendMean.fill(EcalBarrel, double(timestamp_.iLumi), maxEB - minEB);
    meTrendMean.fill(EcalEndcap, double(timestamp_.iLumi), maxEE - minEE);
    meTrendRMS.fill(EcalBarrel, double(timestamp_.iLumi), rmsMaxEB);
    meTrendRMS.fill(EcalEndcap, double(timestamp_.iLumi), rmsMaxEE);
  }

  DEFINE_ECALDQM_WORKER(PresampleClient);
}
