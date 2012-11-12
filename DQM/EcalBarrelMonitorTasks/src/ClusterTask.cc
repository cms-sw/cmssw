#include "../interface/ClusterTask.h"

#include <cmath>

#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace ecaldqm {

  ClusterTask::ClusterTask(edm::ParameterSet const& _workerParams, edm::ParameterSet const& _commonParams) :
    DQWorkerTask(_workerParams, _commonParams, "ClusterTask"),
    topology_(0),
    ebGeometry_(0),
    eeGeometry_(0),
    ebHits_(0),
    eeHits_(0),
    ievt_(0)/*,
    massCalcPrescale_(_workerParams.getUntrackedParameter<int>("massCalcPrescale"))*/
  {
    collectionMask_[kRun] = true;
    collectionMask_[kEBRecHit] = true;
    collectionMask_[kEERecHit] = true;
    collectionMask_[kEBBasicCluster] = true;
    collectionMask_[kEEBasicCluster] = true;
    collectionMask_[kEBSuperCluster] = true;
    collectionMask_[kEESuperCluster] = true;

//     if(massCalcPrescale_ == 0)
//       throw cms::Exception("InvalidConfiguration") << "Mass calculation prescale is zero";
  }

  void
  ClusterTask::setDependencies(DependencySet& _dependencies)
  {
    _dependencies.push_back(Dependency(kEBSuperCluster, kEBRecHit));
    _dependencies.push_back(Dependency(kEESuperCluster, kEERecHit));
  }

  void
  ClusterTask::beginRun(const edm::Run &, const edm::EventSetup &_es)
  {
    edm::ESHandle<CaloTopology> topoHndl;
    _es.get<CaloTopologyRecord>().get(topoHndl);
    topology_ = topoHndl.product();
    if(!topology_)
      throw cms::Exception("EventSetup") << "CaloTopology missing" << std::endl;

    edm::ESHandle<CaloGeometry> geomHndl;
    _es.get<CaloGeometryRecord>().get(geomHndl);
    ebGeometry_ = geomHndl->getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
    eeGeometry_ = geomHndl->getSubdetectorGeometry(DetId::Ecal, EcalEndcap);
    if(!ebGeometry_ || !eeGeometry_)
      throw cms::Exception("EventSetup") << "CaloSubdetectorGeometry missing" << std::endl;

    ievt_ = 0;
  }

  void
  ClusterTask::beginEvent(const edm::Event &, const edm::EventSetup &)
  {
    ebHits_ = 0;
    eeHits_ = 0;

    ievt_++;
  }
  
  bool
  ClusterTask::filterRunType(const std::vector<short>& _runType)
  {
    for(int iFED(0); iFED < 54; iFED++){
      if ( _runType[iFED] == EcalDCCHeaderBlock::COSMIC ||
           _runType[iFED] == EcalDCCHeaderBlock::MTCC ||
           _runType[iFED] == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           _runType[iFED] == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           _runType[iFED] == EcalDCCHeaderBlock::PHYSICS_LOCAL ) return true;
    }

    return false;
  }

  void 
  ClusterTask::runOnRecHits(const EcalRecHitCollection &_hits, Collections _collection)
  {
    switch(_collection){
    case kEBRecHit:
      ebHits_ = &_hits;
      break;
    case kEERecHit:
      eeHits_ = &_hits;
      break;
    default:
      break;
    }
  }

  void
  ClusterTask::runOnBasicClusters(const reco::BasicClusterCollection &_bcs, Collections _collection)
  {
    using namespace std;

    MESet* meBCE(MEs_["BCE"]);
    MESet* meBCEMap(MEs_["BCEMap"]);
    MESet* meBCEMapProjEta(MEs_["BCEMapProjEta"]);
    MESet* meBCEMapProjPhi(MEs_["BCEMapProjPhi"]);
    MESet* meBCOccupancy(MEs_["BCOccupancy"]);
    MESet* meBCOccupancyProjEta(MEs_["BCOccupancyProjEta"]);
    MESet* meBCOccupancyProjPhi(MEs_["BCOccupancyProjPhi"]);
    MESet* meBCSize(MEs_["BCSize"]);
    MESet* meTrendBCSize(online ? MEs_["TrendBCSize"] : 0);
    MESet* meBCSizeMap(MEs_["BCSizeMap"]);
    MESet* meBCSizeMapProjEta(MEs_["BCSizeMapProjEta"]);
    MESet* meBCSizeMapProjPhi(MEs_["BCSizeMapProjPhi"]);
    MESet* meBCNum(MEs_["BCNum"]);
    MESet* meTrendNBC(online ? MEs_["TrendNBC"] : 0);

    int nBC[] = {0, 0};
    bool isBarrel(_collection == kEBBasicCluster);

    //    vector<reco::BasicCluster const*> lowMassCands;

    for(reco::BasicClusterCollection::const_iterator bcItr(_bcs.begin()); bcItr != _bcs.end(); ++bcItr){
      const math::XYZPoint &position(bcItr->position());

      DetId id(bcItr->seed());
      if(id.null()){
	GlobalPoint gp(position.x(), position.y(), position.z());
	const CaloSubdetectorGeometry* subgeom(isBarrel ? ebGeometry_ : eeGeometry_);

	id = subgeom->getClosestCell(gp);
      }

      if(id.null() || (id.subdetId() == EcalBarrel && !isBarrel) || (id.subdetId() == EcalEndcap && isBarrel)) continue;

      float energy(bcItr->energy());

      meBCE->fill(id, energy);

      meBCEMap->fill(id, energy);
      meBCEMapProjEta->fill(position.eta(), energy);
      meBCEMapProjPhi->fill(id, energy);

      meBCOccupancy->fill(id);
      meBCOccupancyProjEta->fill(position.eta());
      meBCOccupancyProjPhi->fill(id);

      float size(bcItr->size());

      meBCSize->fill(id, size);
      if(online) meTrendBCSize->fill(id, double(iLumi), size);

      meBCSizeMap->fill(id, size);
      meBCSizeMapProjEta->fill(position.eta(), size);
      meBCSizeMapProjPhi->fill(id, size);

      int zside(position.z() > 0 ? 1 : 0);
      nBC[zside]++;

//       if(ievt_ % massCalcPrescale_ != 0) continue;

//       if(energy > 10.) continue;

//       EcalRecHitCollection::const_iterator hitItr(isBarrel ? ebHits_->find(id) : eeHits_->find(id));
//       if(hitItr == (isBarrel ? ebHits_->end() : eeHits_->end())) continue;

//       // cuts here must be parametrized
//       if(hitItr->energy() < 0.5) continue;

//       if(hitItr->energy() / energy > 0.95) continue;

//       lowMassCands.push_back(&(*bcItr));
    }

    if(isBarrel){
      meBCNum->fill(unsigned(BinService::kEB + 1), nBC[0] + nBC[1]);
      if(online) meTrendNBC->fill(unsigned(BinService::kEB + 1), double(iLumi), nBC[0] + nBC[1]);
    }else{
      if(meBCNum){
        meBCNum->fill(unsigned(BinService::kEEm + 1), nBC[0]);
        meBCNum->fill(unsigned(BinService::kEEp + 1), nBC[1]);
      }
      if(online) meTrendNBC->fill(unsigned(BinService::kEE + 1), double(iLumi), nBC[0] + nBC[1]);
    }

//     if(ievt_ % massCalcPrescale_ != 0) return;

//     const double pi(3.14159265);

//     for(vector<reco::BasicCluster const*>::iterator bcItr1(lowMassCands.begin()); bcItr1 != lowMassCands.end(); ++bcItr1){
//       reco::BasicCluster const& bc1(**bcItr1);
//       float energy1(bc1.energy());
//       float px1(energy1 * sin(bc1.position().theta()) * cos(bc1.phi()));
//       float py1(energy1 * sin(bc1.position().theta()) * sin(bc1.phi()));
//       float pz1(energy1 * cos(bc1.position().theta()));

//       for(vector<reco::BasicCluster const*>::iterator bcItr2(lowMassCands.begin()); bcItr2 != lowMassCands.end(); ++bcItr2){
// 	if(*bcItr1 == *bcItr2) continue;
// 	reco::BasicCluster const& bc2(**bcItr2);
// 	float energy2(bc2.energy());
// 	float px2(energy2 * sin(bc2.position().theta()) * cos(bc2.phi()));
// 	float py2(energy2 * sin(bc2.position().theta()) * sin(bc2.phi()));
// 	float pz2(energy2 * cos(bc2.position().theta()));

// 	float ptpair(sqrt((px1 + px2) * (px1 + px2) + (py1 + py2) * (py1 + py2)));
// 	if(ptpair < 2.5) continue;

// 	float epair(energy1 + energy2);
// 	float pzpair(abs(pz1 + pz2));

//         float m2(epair * epair - pzpair * pzpair - ptpair * ptpair);
//         if(m2 < 0.) continue;
	
// 	float eta(0.5 * log((epair + pzpair)/(epair - pzpair)));
// 	float phi(atan2(px1 + px2, py1 + py2));

// 	float iso(0.);
// 	for(reco::BasicClusterCollection::const_iterator bcItr(_bcs.begin()); bcItr != _bcs.end(); ++bcItr){
// 	  float dEta(bcItr->eta() - eta);
// 	  float dPhi(bcItr->phi() - phi);
//           if(dPhi > 2. * pi) dPhi -= 2. * pi;
//           else if(dPhi < -2. * pi) dPhi += 2. * pi;
// 	  if(sqrt(dEta * dEta + dPhi * dPhi) < 0.2) iso += bcItr->energy() * sin(bcItr->position().theta());
// 	}
// 	if(iso > 0.5) continue;

// 	float mass(sqrt(m2));
// 	MEs_[kPi0]->fill(mass);
// 	MEs_[kJPsi]->fill(mass);
//       }
//     }
  }

  void
  ClusterTask::runOnSuperClusters(const reco::SuperClusterCollection &_scs, Collections _collection)
  {
    using namespace std;

    MESet* meSCE(MEs_["SCE"]);
    MESet* meSCELow(MEs_["SCELow"]);
    MESet* meSCNBCs(MEs_["SCNBCs"]);
    MESet* meSCNcrystals(MEs_["SCNcrystals"]);
    MESet* meTrendSCSize(online ? MEs_["TrendSCSize"] : 0);
    MESet* meSCSeedEnergy(MEs_["SCSeedEnergy"]);
    MESet* meSCClusterVsSeed(MEs_["SCClusterVsSeed"]);
    MESet* meSCSeedOccupancy(MEs_["SCSeedOccupancy"]);
    MESet* meSingleCrystalCluster(MEs_["SingleCrystalCluster"]);
    MESet* meSCR9(MEs_["SCR9"]);
    MESet* meSCNum(MEs_["SCNum"]);
    MESet* meTrendNSC(online ? MEs_["TrendNSC"] : 0);

    const EcalRecHitCollection *hits(0);
    bool isBarrel;
    if(_collection == kEBSuperCluster){
      hits = ebHits_;
      isBarrel = true;
    }else{
      hits = eeHits_;
      isBarrel = false;
    }

//     reco::SuperCluster const* leading(0);
//     reco::SuperCluster const* subLeading(0);

    int nSC(0);

    for(reco::SuperClusterCollection::const_iterator scItr(_scs.begin()); scItr != _scs.end(); ++scItr){
      const math::XYZPoint &position(scItr->position());

      DetId id(scItr->seed()->seed());
      if(id.null()){
	GlobalPoint gp(position.x(), position.y(), position.z());
	const CaloSubdetectorGeometry* subgeom(isBarrel ? ebGeometry_ : eeGeometry_);

	id = subgeom->getClosestCell(gp);
      }

      if(id.null() || (id.subdetId() == EcalBarrel && !isBarrel) || (id.subdetId() == EcalEndcap && isBarrel)) continue;

      float energy(scItr->energy());

      meSCE->fill(id, energy);
      meSCELow->fill(id, energy);

      meSCNBCs->fill(id, scItr->clustersSize());
      meSCNcrystals->fill(id, scItr->size());

      if(online) meTrendSCSize->fill(id, double(iLumi), scItr->size());

      if(!hits) continue;
      EcalRecHitCollection::const_iterator seedItr(hits->find(id));
      if(seedItr == hits->end()) continue;

      meSCSeedEnergy->fill(id, seedItr->energy());
      meSCClusterVsSeed->fill(id, seedItr->energy(), energy);

      meSCSeedOccupancy->fill(id);

      if(_scs.size() == 1)
        meSingleCrystalCluster->fill(id);

      float e3x3(EcalClusterTools::e3x3(*scItr->seed(), hits, topology_));
      meSCR9->fill(id, e3x3 / energy);

      nSC++;

//       if(ievt_ % massCalcPrescale_ != 0) continue;

//       float et(energy * sin(scItr->position().theta()));
//       if(!leading || et > leading->energy() * sin(leading->position().theta())){
// 	subLeading = leading;
// 	leading = &(*scItr);
//       }
//       else if(!subLeading || et > subLeading->energy() * sin(subLeading->position().theta())){
// 	subLeading = &(*scItr);
//       }
    }

    if(_collection == kEBSuperCluster){
      meSCNum->fill(unsigned(BinService::kEB + 1), nSC);
      if(online) meTrendNSC->fill(unsigned(BinService::kEB + 1), double(iLumi), nSC);
    }
    else{
      meSCNum->fill(unsigned(BinService::kEE + 1), nSC);
      if(online) meTrendNSC->fill(unsigned(BinService::kEE + 1), double(iLumi), nSC);
    }

//     if(ievt_ % massCalcPrescale_ != 0) return;

//     // implement isolation & cuts
//     if(!leading || !subLeading) return;
//     float e(leading->energy() + subLeading->energy());
//     float px(leading->energy() * sin(leading->position().theta()) * cos(leading->phi()) + subLeading->energy() * sin(subLeading->position().theta()) * cos(subLeading->phi()));
//     float py(leading->energy() * sin(leading->position().theta()) * sin(leading->phi()) + subLeading->energy() * sin(subLeading->position().theta()) * sin(subLeading->phi()));
//     float pz(leading->energy() * cos(leading->position().theta()) + subLeading->energy() * cos(subLeading->position().theta()));
//     float m2(e * e - px * px - py * py - pz * pz);
//     if(m2 < 0.) return;
//     float mass(sqrt(m2));
//     MEs_[kZ]->fill(mass);
//     MEs_[kHighMass]->fill(mass);

  }

  DEFINE_ECALDQM_WORKER(ClusterTask);
}

