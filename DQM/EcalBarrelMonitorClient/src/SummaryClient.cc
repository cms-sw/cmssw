#include "../interface/SummaryClient.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  SummaryClient::SummaryClient(edm::ParameterSet const&  _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "SummaryClient")
  {
  }

  void
  SummaryClient::beginRun(const edm::Run &, const edm::EventSetup &)
  {
    for(unsigned iME(0); iME < MEs_.size(); ++iME)
      MEs_[iME]->book();

    MEs_[kQualitySummary]->resetAll(-1.);
    MEs_[kQualitySummary]->reset(1.);
    MEs_[kReportSummaryMap]->resetAll(-1.);
    MEs_[kReportSummaryMap]->reset(1.);
    MEs_[kReportSummaryContents]->reset(1.);
    MEs_[kReportSummary]->reset(1.);
  }

  void
  SummaryClient::producePlots()
  {
    MEs_[kReportSummaryMap]->reset(1.);
    MEs_[kReportSummaryContents]->reset(1.);
    MEs_[kReportSummary]->reset(1.);

    float totalChannels(0.);
    float totalGood(0.);

    std::vector<float> dccChannels(BinService::nDCC, 0.);
    std::vector<float> dccGood(BinService::nDCC, 0.);

    MESet::iterator qEnd(MEs_[kQualitySummary]->end());
    for(MESet::iterator qItr(MEs_[kQualitySummary]->beginChannel()); qItr != qEnd; qItr.toNextChannel()){

      DetId id(qItr->getId());

      float integrity(sources_[kIntegrity]->getBinContent(id));

      if(integrity == 2. || integrity == 5.){
        qItr->setBinContent(integrity);
        continue;
      }

      float presample(sources_[kPresample]->getBinContent(id));
      float timing(sources_[kTiming]->getBinContent(id));
      float rawdata(MEs_[kRawData]->getBinContent(id));

      float status(1.);
      if(integrity == 0. || presample == 0. || timing == 0. || rawdata == 0.)
        status = 0.;
      else if(integrity == 3. || presample == 3. || timing == 3. || rawdata == 3.)
        status = 3.;

      qItr->setBinContent(status);

      unsigned iDCC(dccId(id) - 1);

      if(status == 1.){
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
      MEs_[kReportSummaryMap]->fill(dccid, frac);
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
  }

  DEFINE_ECALDQM_WORKER(SummaryClient);
}

