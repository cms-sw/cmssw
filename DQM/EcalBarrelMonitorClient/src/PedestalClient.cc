#include "../interface/PedestalClient.h"

#include "DQM/EcalCommon/interface/MESetMulti.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iomanip>

namespace ecaldqm
{
  PedestalClient::PedestalClient() :
    DQWorkerClient(),
    gainToME_(),
    pnGainToME_(),
    minChannelEntries_(0),
    expectedMean_(0.),
    toleranceMean_(0.),
    toleranceRMS_(0),
    expectedPNMean_(0.),
    tolerancePNMean_(0.),
    tolerancePNRMS_(0)
  {
  }

  void
  PedestalClient::setParams(edm::ParameterSet const& _params)
  {
    minChannelEntries_ = _params.getUntrackedParameter<int>("minChannelEntries");
    expectedMean_ = _params.getUntrackedParameter<double>("expectedMean");
    toleranceMean_ = _params.getUntrackedParameter<double>("toleranceMean");
    expectedPNMean_ = _params.getUntrackedParameter<double>("expectedPNMean");
    tolerancePNMean_ = _params.getUntrackedParameter<double>("tolerancePNMean");

    std::vector<int> MGPAGains(_params.getUntrackedParameter<std::vector<int> >("MGPAGains"));
    std::vector<int> MGPAGainsPN(_params.getUntrackedParameter<std::vector<int> >("MGPAGainsPN"));

    MESet::PathReplacements repl;

    MESetMulti const& pedestal(static_cast<MESetMulti const&>(sources_.at("Pedestal")));
    unsigned nG(MGPAGains.size());
    for(unsigned iG(0); iG != nG; ++iG){
      int gain(MGPAGains[iG]);
      if(gain != 1 && gain != 6 && gain != 12) throw cms::Exception("InvalidConfiguration") << "MGPA gain";
      repl["gain"] = std::to_string(gain);
      gainToME_[gain] = pedestal.getIndex(repl);
    }

    repl.clear();

    MESetMulti const& pnPedestal(static_cast<MESetMulti const&>(sources_.at("PNPedestal")));
    unsigned nGPN(MGPAGainsPN.size());
    for(unsigned iG(0); iG != nGPN; ++iG){
      int gain(MGPAGainsPN[iG]);
      if(gain != 1 && gain != 16) throw cms::Exception("InvalidConfiguration") << "PN MGPA gain";
      repl["pngain"] = std::to_string(gain);
      pnGainToME_[gain] = pnPedestal.getIndex(repl);
    }

    toleranceRMS_.resize(nG);

    std::vector<double> inToleranceRMS(_params.getUntrackedParameter<std::vector<double> >("toleranceRMS"));

    for(std::map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      unsigned iME(gainItr->second);
      unsigned iGain(0);
      switch(gainItr->first){
      case 1:
        iGain = 0; break;
      case 6:
        iGain = 1; break;
      case 12:
        iGain = 2; break;
      }

      toleranceRMS_[iME] = inToleranceRMS[iGain];
    }

    tolerancePNRMS_.resize(nGPN);

    std::vector<double> inTolerancePNRMS(_params.getUntrackedParameter<std::vector<double> >("tolerancePNRMS"));

    for(std::map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      unsigned iME(gainItr->second);
      unsigned iGain(0);
      switch(gainItr->first){
      case 1:
        iGain = 0; break;
      case 16:
        iGain = 1; break;
      }

      tolerancePNRMS_[iME] = inTolerancePNRMS[iGain];
    }

    qualitySummaries_.insert("Quality");
    qualitySummaries_.insert("QualitySummary");
    qualitySummaries_.insert("PNQualitySummary");
  }

  void
  PedestalClient::producePlots(ProcessType)
  {
    using namespace std;

    MESetMulti& meQuality(static_cast<MESetMulti&>(MEs_.at("Quality")));
    MESetMulti& meQualitySummary(static_cast<MESetMulti&>(MEs_.at("QualitySummary")));
    MESetMulti& meMean(static_cast<MESetMulti&>(MEs_.at("Mean")));
    MESetMulti& meRMS(static_cast<MESetMulti&>(MEs_.at("RMS")));
    MESetMulti& mePNQualitySummary(static_cast<MESetMulti&>(MEs_.at("PNQualitySummary")));
    MESetMulti& mePNRMS(static_cast<MESetMulti&>(MEs_.at("PNRMS")));

    MESetMulti const& sPedestal(static_cast<MESetMulti const&>(sources_.at("Pedestal")));
    MESetMulti const& sPNPedestal(static_cast<MESetMulti const&>(sources_.at("PNPedestal")));

    for(map<int, unsigned>::iterator gainItr(gainToME_.begin()); gainItr != gainToME_.end(); ++gainItr){
      meQuality.use(gainItr->second);
      meQualitySummary.use(gainItr->second);
      meMean.use(gainItr->second);
      meRMS.use(gainItr->second);

      sPedestal.use(gainItr->second);

      uint32_t mask(0);
      switch(gainItr->first){
      case 1:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR);
        break;
      case 6:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_MIDDLE_GAIN_RMS_ERROR);
        break;
      case 12:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR);
        break;
      default:
        break;
      }

      MESet::iterator qEnd(meQuality.end());
      MESet::const_iterator pItr(sPedestal);
      for(MESet::iterator qItr(meQuality.beginChannel()); qItr != qEnd; qItr.toNextChannel()){

        DetId id(qItr->getId());

        bool doMask(meQuality.maskMatches(id, mask, statusManager_));

        pItr = qItr;

        float entries(pItr->getBinEntries());

        if(entries < minChannelEntries_){
          qItr->setBinContent(doMask ? kMUnknown : kUnknown);
          continue;
        }

        float mean(pItr->getBinContent());
        float rms(pItr->getBinError() * sqrt(entries));

        meMean.fill(id, mean);
        meRMS.fill(id, rms);

        if(abs(mean - expectedMean_) > toleranceMean_ || rms > toleranceRMS_[gainItr->second])
          qItr->setBinContent(doMask ? kMBad : kBad);
        else
          qItr->setBinContent(doMask ? kMGood : kGood);
      }

      towerAverage_(meQualitySummary, meQuality, 0.2);
    }

    for(map<int, unsigned>::iterator gainItr(pnGainToME_.begin()); gainItr != pnGainToME_.end(); ++gainItr){
      mePNQualitySummary.use(gainItr->second);
      mePNRMS.use(gainItr->second);

      sPNPedestal.use(gainItr->second);

      uint32_t mask(0);
      switch(gainItr->first){
      case 1:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_LOW_GAIN_RMS_ERROR);
        break;
      case 16:
        mask |= (1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_MEAN_ERROR |
                 1 << EcalDQMStatusHelper::PEDESTAL_HIGH_GAIN_RMS_ERROR);
        break;
      default:
        break;
      }

      for(unsigned iDCC(0); iDCC < nDCC; ++iDCC){

        if(memDCCIndex(iDCC + 1) == unsigned(-1)) continue;

        for(unsigned iPN(0); iPN < 10; ++iPN){
          int subdet(0);
          if(iDCC >= kEBmLow && iDCC <= kEBpHigh) subdet = EcalBarrel;
          else subdet = EcalEndcap;

          EcalPnDiodeDetId id(subdet, iDCC + 1, iPN + 1);

          bool doMask(mePNQualitySummary.maskMatches(id, mask, statusManager_));

          float entries(sPNPedestal.getBinEntries(id));

          if(entries < minChannelEntries_){
            mePNQualitySummary.setBinContent(id, doMask ? kMUnknown : kUnknown);
            continue;
          }

          float mean(sPNPedestal.getBinContent(id));
          float rms(sPNPedestal.getBinError(id) * sqrt(entries));

          mePNRMS.fill(id, rms);

          if(abs(mean - expectedPNMean_) > tolerancePNMean_ || rms > tolerancePNRMS_[gainItr->second])
            mePNQualitySummary.setBinContent(id, doMask ? kMBad : kBad);
          else
            mePNQualitySummary.setBinContent(id, doMask ? kMGood : kGood);
        }
      }
    }
  }

  DEFINE_ECALDQM_WORKER(PedestalClient);
}
