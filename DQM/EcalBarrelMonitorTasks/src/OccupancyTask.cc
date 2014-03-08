#include "../interface/OccupancyTask.h"

#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace ecaldqm
{
  OccupancyTask::OccupancyTask() :
    DQWorkerTask(),
    recHitThreshold_(0.),
    tpThreshold_(0.)
  {
  }

  void
  OccupancyTask::setParams(edm::ParameterSet const& _params)
  {
    recHitThreshold_ = _params.getUntrackedParameter<double>("recHitThreshold");
    tpThreshold_ = _params.getUntrackedParameter<double>("tpThreshold");
  }

  bool
  OccupancyTask::filterRunType(short const* _runType)
  {
    for(int iFED(0); iFED < 54; iFED++){
      if(_runType[iFED] == EcalDCCHeaderBlock::COSMIC ||
	 _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
	 _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
	 _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
	 _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
	 _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL) return true;
    }

    return false;
  }

  void
  OccupancyTask::runOnRawData(EcalRawDataCollection const& _dcchs)
  {
    MESet& meDCC(MEs_.at("DCC"));

    for(EcalRawDataCollection::const_iterator dcchItr(_dcchs.begin()); dcchItr != _dcchs.end(); ++dcchItr)
      meDCC.fill(dcchItr->id());
  }

  template<typename DigiCollection>
  void
  OccupancyTask::runOnDigis(DigiCollection const& _digis, Collections _collection)
  {
    MESet& meDigi(MEs_.at("Digi"));
    MESet& meDigiProjEta(MEs_.at("DigiProjEta"));
    MESet& meDigiProjPhi(MEs_.at("DigiProjPhi"));
    MESet& meDigiAll(MEs_.at("DigiAll"));
    MESet& meDigiDCC(MEs_.at("DigiDCC"));
    MESet& meDigi1D(MEs_.at("Digi1D"));
    MESet& meTrendNDigi(MEs_.at("TrendNDigi"));

    std::for_each(_digis.begin(), _digis.end(), [&](typename DigiCollection::Digi const& digi){
                    DetId id(digi.id());
                    meDigi.fill(id);
                    meDigiProjEta.fill(id);
                    meDigiProjPhi.fill(id);
                    meDigiAll.fill(id);
                    meDigiDCC.fill(id);
                  });

    int iSubdet(_collection == kEBDigi ? EcalBarrel : EcalEndcap);
    meDigi1D.fill(iSubdet, double(_digis.size()));
    meTrendNDigi.fill(iSubdet, double(timestamp_.iLumi), double(_digis.size()));
  }

  void
  OccupancyTask::runOnTPDigis(EcalTrigPrimDigiCollection const& _digis)
  {
    //    MESet& meTPDigiAll(MEs_.at("TPDigiAll"));
    //    MESet& meTPDigiProjEta(MEs_.at("TPDigiProjEta"));
    //    MESet& meTPDigiProjPhi(MEs_.at("TPDigiProjPhi"));
    MESet& meTPDigiThrAll(MEs_.at("TPDigiThrAll"));
    MESet& meTPDigiThrProjEta(MEs_.at("TPDigiThrProjEta"));
    MESet& meTPDigiThrProjPhi(MEs_.at("TPDigiThrProjPhi"));
    MESet& meTrendNTPDigi(MEs_.at("TrendNTPDigi"));

    double nFilteredEB(0.);
    double nFilteredEE(0.);

    std::for_each(_digis.begin(), _digis.end(), [&](EcalTrigPrimDigiCollection::value_type const& digi){
                    EcalTrigTowerDetId const& id(digi.id());
                    //       meTPDigiProjEta.fill(id);
                    //       meTPDigiProjPhi.fill(id);
                    //       meTPDigiAll.fill(id);
                    if(digi.compressedEt() > tpThreshold_){
                      meTPDigiThrProjEta.fill(id);
                      meTPDigiThrProjPhi.fill(id);
                      meTPDigiThrAll.fill(id);
                      if(id.subDet() == EcalBarrel) nFilteredEB += 1.;
                      else nFilteredEE += 1.;
                    }
                  });

    meTrendNTPDigi.fill(EcalBarrel, double(timestamp_.iLumi), nFilteredEB);
    meTrendNTPDigi.fill(EcalEndcap, double(timestamp_.iLumi), nFilteredEE);
  }

  void
  OccupancyTask::runOnRecHits(EcalRecHitCollection const& _hits, Collections _collection)
  {
    MESet& meRecHitAll(MEs_.at("RecHitAll"));
    MESet& meRecHitProjEta(MEs_.at("RecHitProjEta"));
    MESet& meRecHitProjPhi(MEs_.at("RecHitProjPhi"));
    MESet& meRecHitThrAll(MEs_.at("RecHitThrAll"));
    MESet& meRecHitThrProjEta(MEs_.at("RecHitThrProjEta"));
    MESet& meRecHitThrProjPhi(MEs_.at("RecHitThrProjPhi"));
    MESet& meRecHitThr1D(MEs_.at("RecHitThr1D"));
    MESet& meTrendNRecHitThr(MEs_.at("TrendNRecHitThr"));

    uint32_t mask(~(0x1 << EcalRecHit::kGood));
    double nFiltered(0.);

    std::for_each(_hits.begin(), _hits.end(), [&](EcalRecHitCollection::value_type const& hit){
                    DetId id(hit.id());

                    meRecHitAll.fill(id);
                    meRecHitProjEta.fill(id);
                    meRecHitProjPhi.fill(id);

                    if(!hit.checkFlagMask(mask) && hit.energy() > recHitThreshold_){
                      meRecHitThrProjEta.fill(id);
                      meRecHitThrProjPhi.fill(id);
                      meRecHitThrAll.fill(id);
                      nFiltered += 1.;
                    }
                  });

    int iSubdet(_collection == kEBRecHit ? EcalBarrel : EcalEndcap);
    meRecHitThr1D.fill(iSubdet, nFiltered);
    meTrendNRecHitThr.fill(iSubdet, double(timestamp_.iLumi), nFiltered);
  }

  DEFINE_ECALDQM_WORKER(OccupancyTask);
}

