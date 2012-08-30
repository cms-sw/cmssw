#include "../interface/SummaryClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  SummaryClient::SummaryClient(edm::ParameterSet const&  _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "SummaryClient")
  {
  }

  void
  SummaryClient::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    MEs_[kQualitySummary]->resetAll(-1.);
    MEs_[kQualitySummary]->reset(kUnknown);
    MEs_[kReportSummaryMap]->resetAll(-1.);
    MEs_[kReportSummaryMap]->reset(1.);
    MEs_[kReportSummaryContents]->reset(1.);
    MEs_[kReportSummary]->reset(1.);
  }

  void
  SummaryClient::producePlots()
  {
    float totalChannels(0.);
    float totalGood(0.);

    std::vector<float> dccChannels(BinService::nDCC, 0.);
    std::vector<float> dccGood(BinService::nDCC, 0.);

    MESet::iterator qEnd(MEs_[kQualitySummary]->end());
    MESet::const_iterator iItr(sources_[kIntegrity]);
    MESet::const_iterator pItr(sources_[kPresample]);
    MESet::const_iterator hItr(sources_[kHotCell]);
    for(MESet::iterator qItr(MEs_[kQualitySummary]->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      DetId id(qItr->getId());

      iItr = qItr;

      int integrity(iItr->getBinContent());

      if(integrity == kUnknown){
        qItr->setBinContent(integrity);
        continue;
      }

      pItr = qItr;
      hItr = qItr;

      int presample(pItr->getBinContent());
      int hotcell(hItr->getBinContent());
      int timing(sources_[kTiming]->getBinContent(id));
      int rawdata(sources_[kRawData]->getBinContent(id));
      int trigprim(sources_[kTriggerPrimitives]->getBinContent(id));

      int status(kGood);
      if(integrity == kBad || presample == kBad || timing == kBad || rawdata == kBad || trigprim == kBad || hotcell == kBad)
        status = kBad;
      else if(integrity == kUnknown && presample == kUnknown && timing == kUnknown && rawdata == kUnknown && trigprim == kUnknown)
        status = kUnknown;

      qItr->setBinContent(status);

      unsigned iDCC(dccId(id) - 1);

      if(status != kBad){
        dccGood[iDCC] += 1.;
        totalGood += 1.;
      }
      dccChannels[iDCC] += 1.;
      totalChannels += 1.;
    }

    for(unsigned iDCC(0); iDCC < BinService::nDCC; ++iDCC){
      if(dccChannels[iDCC] < 1.) continue;

      unsigned dccid(iDCC + 1);
      float frac(dccGood[iDCC] / dccChannels[iDCC]);
      MEs_[kReportSummaryMap]->setBinContent(dccid, frac);
      MEs_[kReportSummaryContents]->fill(dccid, frac);
    }

    if(totalChannels > 0.) MEs_[kReportSummary]->fill(totalGood / totalChannels);
  }

  /*static*/
  void
  SummaryClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["QualitySummary"] = kQualitySummary;
    _nameToIndex["ReportSummaryMap"] = kReportSummaryMap;
    _nameToIndex["ReportSummaryContents"] = kReportSummaryContents;
    _nameToIndex["ReportSummary"] = kReportSummary;

    _nameToIndex["Integrity"] = kIntegrity;
    _nameToIndex["Presample"] = kPresample;
    _nameToIndex["Timing"] = kTiming;
    _nameToIndex["RawData"] = kRawData;
    _nameToIndex["TriggerPrimitives"] = kTriggerPrimitives;
    _nameToIndex["HotCell"] = kHotCell;
  }

  DEFINE_ECALDQM_WORKER(SummaryClient);
}

