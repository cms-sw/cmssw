#include "../interface/OccupancyClient.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DQM/EcalBarrelMonitorTasks/interface/OccupancyTask.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  OccupancyClient::OccupancyClient(const edm::ParameterSet& _params, const edm::ParameterSet& _paths) :
    DQWorkerClient(_params, _paths, "OccupancyClient"),
    geometry_(0),
    minHits_(0),
    deviationThreshold_(0.)
  {
    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));
    minHits_ = taskParams.getUntrackedParameter<int>("minHits");
    deviationThreshold_ = taskParams.getUntrackedParameter<double>("deviationThreshold");

    edm::ParameterSet const& sources(_params.getUntrackedParameterSet("sources"));
    source_(sDigi, "OccupancyTask", OccupancyTask::kDigi, sources);
    source_(sRecHitThr, "OccupancyTask", OccupancyTask::kRecHitThr, sources);
    source_(sTPDigiThr, "OccupancyTask", OccupancyTask::kTPDigiThr, sources);
  }

  void
  OccupancyClient::beginRun(const edm::Run &, const edm::EventSetup &_es)
  {
    edm::ESHandle<CaloGeometry> geomHndl;
    _es.get<CaloGeometryRecord>().get(geomHndl);
    geometry_ = geomHndl.product();
    if(!geometry_)
      throw cms::Exception("EventSetup") << "CaloGeometry invalid";
  }

  void
  OccupancyClient::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kQualitySummary]->resetAll(-1.);
  }

  void
  OccupancyClient::producePlots()
  {
    using namespace std;

    MEs_[kHotDigi]->reset();
    MEs_[kHotRecHitThr]->reset();
    MEs_[kHotTPDigiThr]->reset();

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR);

    vector<double> digiPhiRingMean(28, 0.);
    vector<double> rechitPhiRingMean(28, 0.);
    vector<int> numCrystals(28, 0); // this is static, but is easier to count now

    for(unsigned dccid(1); dccid <= 54; dccid++){
      for(unsigned tower(1); tower <= getNSuperCrystals(dccid); tower++){
	vector<DetId> ids(getElectronicsMap()->dccTowerConstituents(dccid, tower));

	if(ids.size() == 0) continue;

	for(vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
	  float entries(sources_[sDigi]->getBinContent(*idItr));
	  float rhentries(sources_[sRecHitThr]->getBinContent(*idItr));

	  int ieta(getTrigTowerMap()->towerOf(*idItr).ietaAbs());
	  digiPhiRingMean.at(ieta - 1) += entries;
	  rechitPhiRingMean.at(ieta - 1) += rhentries;

	  numCrystals.at(ieta - 1) += 1;
	}
      }
    }

    for(int ie(0); ie < 28; ie++){
      digiPhiRingMean[ie] /= numCrystals[ie];
      rechitPhiRingMean[ie] /= numCrystals[ie];
    }

    // second round to find hot towers
    for(unsigned dccid(1); dccid <= 54; dccid++){
      for(unsigned tower(1); tower <= getNSuperCrystals(dccid); tower++){
	vector<DetId> ids(getElectronicsMap()->dccTowerConstituents(dccid, tower));

	if(ids.size() == 0) continue;

	float quality(1.);
	for(vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
	  float entries(sources_[sDigi]->getBinContent(*idItr));
	  float rhentries(sources_[sRecHitThr]->getBinContent(*idItr));

	  int ieta(getTrigTowerMap()->towerOf(*idItr).ietaAbs());

	  if(entries > minHits_ && entries > digiPhiRingMean.at(ieta - 1) * deviationThreshold_){
	    MEs_[kHotDigi]->fill(*idItr);
	    quality = 0.;
	  }
	  if(rhentries > minHits_ && rhentries > rechitPhiRingMean.at(ieta - 1) * deviationThreshold_){
	    MEs_[kHotRecHitThr]->fill(*idItr);
	    quality = 0.;
	  }
	}
	if(dccid <= 9 || dccid >= 46){
	  vector<EcalScDetId> scs(getElectronicsMap()->getEcalScDetId(dccid, tower));
	  for(vector<EcalScDetId>::iterator scItr(scs.begin()); scItr != scs.end(); ++scItr)
	    fillQuality_(kQualitySummary, *scItr, mask, quality);
	}
	else
	  fillQuality_(kQualitySummary, ids[0], mask, quality);
      }
    }

    vector<double> tpdigiPhiRingMean(28, 0.);

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      float entries(sources_[sTPDigiThr]->getBinContent(ttid));

      tpdigiPhiRingMean.at(ttid.ietaAbs() - 1) += entries;
    }

    for(int ie(0); ie < 28; ie++){
      float denom(-1.);
      if(ie < 27) denom = 72.;
      else denom = 36.;
      tpdigiPhiRingMean[ie] /= denom;
    }

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; iTT++){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      float entries(sources_[sTPDigiThr]->getBinContent(ttid));

      if(entries > minHits_ && entries > tpdigiPhiRingMean.at(ttid.ietaAbs() - 1) * deviationThreshold_){
	MEs_[kHotTPDigiThr]->fill(ttid);
	vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
	for(vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
	  if(MEs_[kQualitySummary]->getBinContent(*idItr) > 0.)
	    fillQuality_(kQualitySummary, *idItr, mask, 0.);
	}
      }   
    }

  }

  /*static*/
  void
  OccupancyClient::setMEData(std::vector<MEData>& _data)
  {
    _data[kHotDigi] = MEData("HotDigi", BinService::kChannel, BinService::kCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kHotRecHitThr] = MEData("HotRecHitThr", BinService::kChannel, BinService::kCrystal, MonitorElement::DQM_KIND_TH1F);
    _data[kHotTPDigiThr] = MEData("HotTPDigiThr", BinService::kChannel, BinService::kTriggerTower, MonitorElement::DQM_KIND_TH1F);
    _data[kQualitySummary] = MEData("QualitySummary", BinService::kEcal2P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
  }

  DEFINE_ECALDQM_WORKER(OccupancyClient);
}

