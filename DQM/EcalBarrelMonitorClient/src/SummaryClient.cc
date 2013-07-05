#include "../interface/SummaryClient.h"

#include "DQM/EcalBarrelMonitorClient/interface/IntegrityClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/PresampleClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/TimingClient.h"
#include "DQM/EcalBarrelMonitorClient/interface/RawDataClient.h"
#include "DQM/EcalBarrelMonitorTasks/interface/OccupancyTask.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  SummaryClient::SummaryClient(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerClient(_params, _paths, "SummaryClient")
  {
    edm::ParameterSet const& sources(_params.getUntrackedParameterSet("sources"));
    source_(sIntegrity, "IntegrityClient", IntegrityClient::kQuality, sources);
    source_(sPresample, "PresampleClient", PresampleClient::kQuality, sources);
    source_(sTiming, "TimingClient", TimingClient::kQuality, sources);
    source_(sRawData, "RawDataClient", RawDataClient::kQualitySummary, sources);
    source_(sDigiOccupancy, "OccupancyTask", OccupancyTask::kDigi, sources);
  }

  void
  SummaryClient::bookMEs()
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
    MEs_[kQualitySummary]->reset(2.);
    MEs_[kReportSummaryMap]->reset(1.);

    float totalChannels(0.);
    float totalGood(0.);

    for(unsigned dccid(1); dccid <= 54; dccid++){
      MonitorElement const* me(sources_[sDigiOccupancy]->getME(dccid - 1));
      if(me && me->getTH1()->GetEntries() < 1.){
	MEs_[kReportSummaryContents]->fill(dccid, 1.);
	continue;
      }

      float dccChannels(0.);
      float dccGood(0.);

      for(unsigned tower(1); tower <= getNSuperCrystals(dccid); tower++){
	std::vector<DetId> ids(getElectronicsMap()->dccTowerConstituents(dccid, tower));

	if(ids.size() == 0) continue;

	float towerChannels(0.);
	float towerGood(0.);

	for(std::vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){

	  float integrity(sources_[sIntegrity]->getBinContent(*idItr));

	  if(integrity == 2.) continue;

	  float presample(sources_[sPresample]->getBinContent(*idItr));
	  float timing(sources_[sTiming]->getBinContent(*idItr));
	  float rawdata(sources_[sRawData]->getBinContent(*idItr));

	  if(integrity > 2.) integrity = 1.;
	  if(presample > 2.) presample = 1.;
	  if(timing > 2.) timing = 1.;
	  if(rawdata > 2.) rawdata = 1.;

	  float status(1.);
	  if(integrity < 2.) status *= integrity;
	  if(presample < 2.) status *= presample;
	  if(timing < 2.) status *= timing;
	  if(rawdata < 2.) status *= rawdata;

	  MEs_[kQualitySummary]->setBinContent(*idItr, status);

	  if(status == 1.){
	    towerGood += 1.;
	    dccGood += 1.;
	    totalGood += 1.;
	  }
	  towerChannels += 1.;
	  dccChannels += 1.;
	  totalChannels += 1.;
	}

	if(towerChannels < 1.) continue;

	if(dccid <= 9 || dccid >= 46){
	  std::vector<EcalScDetId> scs(getElectronicsMap()->getEcalScDetId(dccid, tower));
	  for(std::vector<EcalScDetId>::iterator scItr(scs.begin()); scItr != scs.end(); ++scItr)
	    MEs_[kReportSummaryMap]->setBinContent(*scItr, towerGood / towerChannels);
	}
	else
	  MEs_[kReportSummaryMap]->setBinContent(ids[0], towerGood / towerChannels);
      }

      if(dccChannels < 1.) continue;

      MEs_[kReportSummaryContents]->fill(dccid, dccGood / dccChannels);
    }

    if(totalChannels > 0.) MEs_[kReportSummary]->fill(totalGood / totalChannels);

  }

  /*static*/
  void
  SummaryClient::setMEData(std::vector<MEData>& _data)
  {
    _data[kQualitySummary] = MEData("QualitySummary", BinService::kEcal2P, BinService::kCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kReportSummaryMap] = MEData("ReportSummaryMap", BinService::kEcal, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kReportSummaryContents] = MEData("ReportSummaryContents", BinService::kSM, BinService::kReport, MonitorElement::DQM_KIND_REAL);
    _data[kReportSummary] = MEData("ReportSummary", BinService::kEcal, BinService::kReport, MonitorElement::DQM_KIND_REAL);
  }

  DEFINE_ECALDQM_WORKER(SummaryClient);
}

