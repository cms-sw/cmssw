#include "../interface/RawDataClient.h"

#include "DQM/EcalBarrelMonitorTasks/interface/RawDataTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/FEFlags.h"

#include <cmath>

namespace ecaldqm {

  RawDataClient::RawDataClient(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerClient(_params, _paths, "RawDataClient"),
    synchErrorThreshold_(0)
  {
    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));
    synchErrorThreshold_ = taskParams.getUntrackedParameter<int>("synchErrorThreshold");

    edm::ParameterSet const& sources(_params.getUntrackedParameterSet("sources"));
    source_(sL1ADCC, "RawDataTask", RawDataTask::kL1ADCC, sources);
    source_(sFEStatus, "RawDataTask", RawDataTask::kFEStatus, sources);
  }

  void
  RawDataClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kQualitySummary]->resetAll(-1.);
  }

  void
  RawDataClient::producePlots()
  {
    uint32_t mask(1 << EcalDQMStatusHelper::STATUS_FLAG_ERROR);

    for(unsigned dccid(1); dccid <= 54; dccid++){

      float l1aDesync(sources_[sL1ADCC]->getBinContent(dccid));

      float dccStatus(l1aDesync > synchErrorThreshold_ ? 0. : 1.);

      for(unsigned tower(1); tower <= getNSuperCrystals(dccid); tower++){
	std::vector<DetId> ids(getElectronicsMap()->dccTowerConstituents(dccid, tower));

	if(ids.size() == 0) continue;

	float towerStatus(dccStatus);

	if(towerStatus > 0.){ // if the DCC is good, look into individual FEs
	  float towerEntries(0.);
	  for(unsigned iS(0); iS < nFEFlags; iS++){
	    float entries(sources_[sFEStatus]->getBinContent(ids[0], iS + 1));
	    towerEntries += entries;
	    if(entries > 0. &&
	       iS != Enabled && iS != Disabled && iS != Suppressed &&
	       iS != FIFOFull && iS != FIFOFullL1ADesync && iS != ForcedZS)
	      towerStatus = 0.;
	  }

	  if(towerEntries < 1.) towerStatus = 2.;
	}

	if(dccid <= 9 || dccid >= 46){
	  std::vector<EcalScDetId> scs(getElectronicsMap()->getEcalScDetId(dccid, tower));
	  for(std::vector<EcalScDetId>::iterator scItr(scs.begin()); scItr != scs.end(); ++scItr)
	    fillQuality_(kQualitySummary, *scItr, mask, towerStatus);
	}
	else
	  fillQuality_(kQualitySummary, ids[0], mask, towerStatus);
      }
    }
  }

  /*static*/
  void
  RawDataClient::setMEData(std::vector<MEData>& _data)
  {
    _data[kQualitySummary] = MEData("QualitySummary", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
  }

  DEFINE_ECALDQM_WORKER(RawDataClient);
}

