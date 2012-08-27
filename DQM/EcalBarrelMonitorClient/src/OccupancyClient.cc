#include "../interface/OccupancyClient.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "CondFormats/EcalObjects/interface/EcalDQMStatusHelper.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  OccupancyClient::OccupancyClient(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerClient(_workerParams, _commonParams, "OccupancyClient"),
    geometry_(0),
    minHits_(_workerParams.getUntrackedParameter<int>("minHits")),
    deviationThreshold_(_workerParams.getUntrackedParameter<double>("deviationThreshold"))
  {
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

    uint32_t mask(1 << EcalDQMStatusHelper::PEDESTAL_ONLINE_HIGH_GAIN_RMS_ERROR |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_WARNING |
                  1 << EcalDQMStatusHelper::PHYSICS_BAD_CHANNEL_ERROR);

    vector<double> digiPhiRingMean(56, 0.);
    vector<double> rechitPhiRingMean(56, 0.);
    vector<int> numCrystals(56, 0); // this is static, but is easier to count now

    MESet::const_iterator dEnd(sources_[kDigi]->end());
    MESet::const_iterator rItr(sources_[kRecHitThr]);
    for(MESet::const_iterator dItr(sources_[kDigi]->beginChannel()); dItr != dEnd; dItr.toNextChannel()){
      rItr = dItr;

      float entries(dItr->getBinContent());
      float rhentries(rItr->getBinContent());

      int ieta(getTrigTowerMap()->towerOf(dItr->getId()).ieta());
      unsigned index(ieta < 0 ? ieta + 28 : ieta + 27);

      digiPhiRingMean.at(index) += entries;
      rechitPhiRingMean.at(index) += rhentries;
      numCrystals.at(index) += 1;
    }

    for(int ie(0); ie < 56; ie++){
      digiPhiRingMean[ie] /= numCrystals[ie];
      rechitPhiRingMean[ie] /= numCrystals[ie];
    }

    // second round to find hot towers
    for(MESet::const_iterator dItr(sources_[kDigi]->beginChannel()); dItr != dEnd; dItr.toNextChannel()){
      rItr = dItr;

      DetId id(dItr->getId());

      float entries(dItr->getBinContent());
      float rhentries(rItr->getBinContent());

      int ieta(getTrigTowerMap()->towerOf(dItr->getId()).ieta());
      unsigned index(ieta < 0 ? ieta + 28 : ieta + 27);

      if(entries > minHits_){
        if(entries > digiPhiRingMean.at(index) * deviationThreshold_){
          MEs_[kHotDigi]->fill(id);
          MEs_[kQualitySummary]->setBinContent(id, maskQuality_(kQualitySummary, id, mask, 0));
          continue;
        }
        
        if(rhentries > minHits_){
          if(rhentries > rechitPhiRingMean.at(index) * deviationThreshold_){
            MEs_[kHotRecHitThr]->fill(id);
            MEs_[kQualitySummary]->setBinContent(id, maskQuality_(kQualitySummary, id, mask, 0));
            continue;
          }

	  MEs_[kQualitySummary]->setBinContent(id, maskQuality_(kQualitySummary, id, mask, 1));
        }
      }
      else
        MEs_[kQualitySummary]->setBinContent(id, maskQuality_(kQualitySummary, id, mask, 2));
    }

    vector<double> tpdigiPhiRingMean(56, 0.);

    print_("determing phi-ring tower mean");

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; ++iTT){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      float entries(sources_[kTPDigiThr]->getBinContent(ttid));

      unsigned index(ttid.ieta() < 0 ? ttid.ieta() + 28 : ttid.ieta() + 27);
      tpdigiPhiRingMean.at(index) += entries;
    }

    for(int ie(0); ie < 28; ie++){
      float denom(-1.);
      if(ie < 27) denom = 72.;
      else denom = 36.;
      tpdigiPhiRingMean[ie] /= denom;
      tpdigiPhiRingMean[55 - ie] /= denom;
    }

    print_("detecting hot towers");

    for(unsigned iTT(0); iTT < EcalTrigTowerDetId::kSizeForDenseIndexing; ++iTT){
      EcalTrigTowerDetId ttid(EcalTrigTowerDetId::detIdFromDenseIndex(iTT));
      float entries(sources_[kTPDigiThr]->getBinContent(ttid));

      if(entries > minHits_){
        if(entries > tpdigiPhiRingMean.at(ttid.ietaAbs() - 1) * deviationThreshold_){
          MEs_[kHotTPDigiThr]->fill(ttid);
          MEs_[kQualitySummary]->setBinContent(ttid, maskQuality_(kQualitySummary, ttid, mask, 0));
          continue;
        }

        MEs_[kQualitySummary]->setBinContent(ttid, maskQuality_(kQualitySummary, ttid, mask, 1));
      }
      else
        MEs_[kQualitySummary]->setBinContent(ttid, maskQuality_(kQualitySummary, ttid, mask, 2));
    }
  }

  /*static*/
  void
  OccupancyClient::setMEOrdering(std::map<std::string, unsigned>& _nameToIndex)
  {
    _nameToIndex["HotDigi"] = kHotDigi;
    _nameToIndex["HotRecHitThr"] = kHotRecHitThr;
    _nameToIndex["HotTPDigiThr"] = kHotTPDigiThr;
    _nameToIndex["QualitySummary"] = kQualitySummary;

    _nameToIndex["Digi"] = kDigi;
    _nameToIndex["RecHitThr"] = kRecHitThr;
    _nameToIndex["TPDigiThr"] = kTPDigiThr;
  }

  DEFINE_ECALDQM_WORKER(OccupancyClient);
}
