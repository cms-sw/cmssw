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

  ClusterTask::ClusterTask(const edm::ParameterSet &_params, const::edm::ParameterSet& _paths) :
    DQWorkerTask(_params, _paths, "ClusterTask"),
    topology_(0),
    ebGeometry_(0),
    eeGeometry_(0),
    ebHits_(0),
    eeHits_(0),
    ievt_(0),
    lowEMax_(0.),
    massCalcPrescale_(0)
  {
    collectionMask_ = 
      (0x1 << kRun) |
      (0x1 << kEBRecHit) |
      (0x1 << kEERecHit) |
      (0x1 << kEBBasicCluster) |
      (0x1 << kEEBasicCluster) |
      (0x1 << kEBSuperCluster) |
      (0x1 << kEESuperCluster);

    dependencies_.push_back(std::pair<Collections, Collections>(kEBSuperCluster, kEBRecHit));
    dependencies_.push_back(std::pair<Collections, Collections>(kEESuperCluster, kEERecHit));

    edm::ParameterSet const& taskParams(_params.getUntrackedParameterSet(name_));

    lowEMax_ = taskParams.getUntrackedParameter<double>("lowEMax");
    massCalcPrescale_ = taskParams.getUntrackedParameter<int>("massCalcPrescale");
  }

  ClusterTask::~ClusterTask()
  {
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

  void
  ClusterTask::bookMEs()
  {
    DQWorker::bookMEs();

    MEs_[kBCE]->setAxisTitle("energy (GeV)", 1);
    MEs_[kBCNum]->setAxisTitle("number of clusters", 1);
    MEs_[kBCSize]->setAxisTitle("number of clusters", 1);
    MEs_[kSCE]->setAxisTitle("energy (GeV)", 1);
    MEs_[kSCELow]->setAxisTitle("energy (GeV)", 1);
    MEs_[kSCSeedEnergy]->setAxisTitle("energy (GeV)", 1);
    MEs_[kSCClusterVsSeed]->setAxisTitle("seed crystal energy (GeV)", 1);
    MEs_[kSCClusterVsSeed]->setAxisTitle("cluster energy (GeV)", 2);
    MEs_[kSCNum]->setAxisTitle("number of clusters", 1);
    MEs_[kSCNBCs]->setAxisTitle("cluster size", 1);
    MEs_[kSCNcrystals]->setAxisTitle("cluster size", 1);
    MEs_[kSCR9]->setAxisTitle("R9", 1);
    MEs_[kPi0]->setAxisTitle("mass (GeV)", 1);
    MEs_[kJPsi]->setAxisTitle("mass (GeV)", 1);
    MEs_[kZ]->setAxisTitle("mass (GeV)", 1);
    MEs_[kHighMass]->setAxisTitle("mass (GeV)", 1);

    for(int i(0); i < 2; i++)
      MEs_[kSCELow]->getME(i)->getTH1()->GetXaxis()->SetLimits(0., lowEMax_);
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

    int nBC[] = {0, 0};
    bool isBarrel(_collection == kEBBasicCluster);

    vector<reco::BasicCluster const*> lowMassCands;

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

      MEs_[kBCE]->fill(id, energy);

      MEs_[kBCEMap]->fill(id, energy);
      MEs_[kBCEMapProjEta]->fill(id, energy);
      MEs_[kBCEMapProjPhi]->fill(id, energy);

      MEs_[kBCOccupancy]->fill(id);
      MEs_[kBCOccupancyProjEta]->fill(id);
      MEs_[kBCOccupancyProjPhi]->fill(id);

      float size(bcItr->size());

      MEs_[kBCSize]->fill(id, size);

      MEs_[kBCSizeMap]->fill(id, size);
      MEs_[kBCSizeMapProjEta]->fill(id, size);
      MEs_[kBCSizeMapProjPhi]->fill(id, size);

      int zside(position.z() > 0 ? 1 : 0);
      nBC[zside]++;

      if(ievt_ % massCalcPrescale_ != 0) continue;

      if(energy > 10.) continue;

      EcalRecHitCollection::const_iterator hitItr(isBarrel ? ebHits_->find(id) : eeHits_->find(id));
      if(hitItr == (isBarrel ? ebHits_->end() : eeHits_->end())) continue;

      // cuts here must be parametrized
      if(hitItr->energy() < 0.5) continue;

      if(hitItr->energy() / energy > 0.95) continue;

      lowMassCands.push_back(&(*bcItr));
    }

    if(isBarrel){
      MEs_[kBCNum]->fill((unsigned)BinService::kEB + 1, nBC[0] + nBC[1]);
    }else{
      MEs_[kBCNum]->fill((unsigned)BinService::kEEm + 1, nBC[0]);
      MEs_[kBCNum]->fill((unsigned)BinService::kEEp + 1, nBC[1]);
    }

    if(ievt_ % massCalcPrescale_ != 0) return;

    for(vector<reco::BasicCluster const*>::iterator bcItr1(lowMassCands.begin()); bcItr1 != lowMassCands.end(); ++bcItr1){
      reco::BasicCluster const& bc1(**bcItr1);
      float energy1(bc1.energy());
      float px1(energy1 * sin(bc1.position().theta()) * cos(bc1.phi()));
      float py1(energy1 * sin(bc1.position().theta()) * sin(bc1.phi()));
      float pz1(energy1 * cos(bc1.position().theta()));

      for(vector<reco::BasicCluster const*>::iterator bcItr2(lowMassCands.begin()); bcItr2 != lowMassCands.end(); ++bcItr2){
	if(*bcItr1 == *bcItr2) continue;
	reco::BasicCluster const& bc2(**bcItr2);
	float energy2(bc2.energy());
	float px2(energy2 * sin(bc2.position().theta()) * cos(bc2.phi()));
	float py2(energy2 * sin(bc2.position().theta()) * sin(bc2.phi()));
	float pz2(energy2 * cos(bc2.position().theta()));

	float ptpair(sqrt((px1 + px2) * (px1 + px2) + (py1 + py2) * (py1 + py2)));
	if(ptpair < 2.5) continue;

	float epair(energy1 + energy2);
	float pzpair(abs(pz1 + pz2));
	
	if(epair < pzpair + 1.e-10) continue;
	float eta(0.5 * log((epair + pzpair)/(epair - pzpair)));
	float phi(atan2(px1 + px2, py1 + py2));

	float iso(0.);
	for(reco::BasicClusterCollection::const_iterator bcItr(_bcs.begin()); bcItr != _bcs.end(); ++bcItr){
	  float dEta(bcItr->eta() - eta);
	  float dPhi(bcItr->phi() - phi);
	  if(sqrt(dEta * dEta + dPhi * dPhi) < 0.2) iso += bcItr->energy() * sin(bcItr->position().theta());
	}
	if(iso > 0.5) continue;

	float mass(sqrt(epair * epair - pzpair * pzpair - ptpair * ptpair));
	MEs_[kPi0]->fill(mass);
	MEs_[kJPsi]->fill(mass);
      }
    }
  }

  void
  ClusterTask::runOnSuperClusters(const reco::SuperClusterCollection &_scs, Collections _collection)
  {
    using namespace std;

    const EcalRecHitCollection *hits(0);
    bool isBarrel;
    if(_collection == kEBSuperCluster){
      hits = ebHits_;
      isBarrel = true;
    }else{
      hits = eeHits_;
      isBarrel = false;
    }

    reco::SuperCluster const* leading(0);
    reco::SuperCluster const* subLeading(0);

    int nSC[] = {0, 0};

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

      MEs_[kSCE]->fill(id, energy);
      if(energy < lowEMax_) MEs_[kSCELow]->fill(id, energy);

      MEs_[kSCNBCs]->fill(id, scItr->clustersSize());
      MEs_[kSCNcrystals]->fill(id, scItr->size());

      if(!hits) continue;
      EcalRecHitCollection::const_iterator seedItr(hits->find(id));
      if(seedItr == hits->end()) continue;

      MEs_[kSCSeedEnergy]->fill(id, seedItr->energy());
      MEs_[kSCClusterVsSeed]->fill(id, seedItr->energy(), energy);

      MEs_[kSCSeedOccupancy]->fill(id);

      if(_scs.size() == 1)
	MEs_[kSingleCrystalCluster]->fill(id);

      float e3x3(EcalClusterTools::e3x3(*scItr->seed(), hits, topology_));
      MEs_[kSCR9]->fill(id, e3x3 / energy);

      int zside(position.z() > 0 ? 1 : 0);
      nSC[zside]++;

      if(ievt_ % massCalcPrescale_ != 0) continue;

      float et(energy * sin(scItr->position().theta()));
      if(!leading || et > leading->energy() * sin(leading->position().theta())){
	subLeading = leading;
	leading = &(*scItr);
      }
      else if(!subLeading || et > subLeading->energy() * sin(subLeading->position().theta())){
	subLeading = &(*scItr);
      }
    }

    if(_collection == kEBSuperCluster){
      MEs_[kSCNum]->fill((unsigned)BinService::kEB + 1, nSC[0] + nSC[1]);
    }else{
      MEs_[kSCNum]->fill((unsigned)BinService::kEEm + 1, nSC[0]);
      MEs_[kSCNum]->fill((unsigned)BinService::kEEp + 1, nSC[1]);
    }

    if(ievt_ % massCalcPrescale_ != 0) return;

    // implement isolation & cuts
    if(!leading || !subLeading) return;
    float e(leading->energy() + subLeading->energy());
    float px(leading->energy() * sin(leading->position().theta()) * cos(leading->phi()) + subLeading->energy() * sin(subLeading->position().theta()) * cos(subLeading->phi()));
    float py(leading->energy() * sin(leading->position().theta()) * sin(leading->phi()) + subLeading->energy() * sin(subLeading->position().theta()) * sin(subLeading->phi()));
    float pz(leading->energy() * cos(leading->position().theta()) + subLeading->energy() * cos(subLeading->position().theta()));
    float mass(sqrt(e * e - px * px - py * py - pz * pz));
    MEs_[kZ]->fill(mass);
    MEs_[kHighMass]->fill(mass);

  }

  /*static*/
  void
  ClusterTask::setMEData(std::vector<MEData>& _data)
  {
    BinService::AxisSpecs xaxis, yaxis, zaxis;

    zaxis.low = 0.;
    zaxis.high = 50.;
    _data[kBCEMap] = MEData("BCEMap", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &zaxis);
    _data[kBCEMapProjEta] = MEData("BCEMap", BinService::kEcal3P, BinService::kProjEta, MonitorElement::DQM_KIND_TPROFILE);
    _data[kBCEMapProjPhi] = MEData("BCEMap", BinService::kEcal3P, BinService::kProjPhi, MonitorElement::DQM_KIND_TPROFILE);

    _data[kBCOccupancy] = MEData("BCOccupancy", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kBCOccupancyProjEta] = MEData("BCOccupancy", BinService::kEcal3P, BinService::kProjEta, MonitorElement::DQM_KIND_TH1F);
    _data[kBCOccupancyProjPhi] = MEData("BCOccupancy", BinService::kEcal3P, BinService::kProjPhi, MonitorElement::DQM_KIND_TH1F);

    zaxis.high = 30.;
    _data[kBCSizeMap] = MEData("BCSizeMap", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TPROFILE2D, 0, 0, &zaxis);
    _data[kBCSizeMapProjEta] = MEData("BCSizeMap", BinService::kEcal3P, BinService::kProjEta, MonitorElement::DQM_KIND_TPROFILE);
    _data[kBCSizeMapProjPhi] = MEData("BCSizeMap", BinService::kEcal3P, BinService::kProjPhi, MonitorElement::DQM_KIND_TPROFILE);

    xaxis.nbins = 50;
    xaxis.low = 0.;
    xaxis.high = 150.;
    _data[kBCE] = MEData("BCE", BinService::kEcal3P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);

    xaxis.nbins = 20;
    xaxis.low = 0.;
    xaxis.high = 100.;
    _data[kBCNum] = MEData("BCNum", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);

    xaxis.nbins = 50;
    xaxis.low = 0.;
    xaxis.high = 100.;
    _data[kBCSize] = MEData("BCSize", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);

    xaxis.nbins = 50;
    xaxis.low = 0.;
    xaxis.high = 150.;
    _data[kSCE] = MEData("SCE", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);

    xaxis.nbins = 50;
    xaxis.low = 0.;
    xaxis.high = 10.;
    _data[kSCELow] = MEData("SCELow", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);

    xaxis.nbins = 50;
    xaxis.low = 0.;
    xaxis.high = 150.;
    _data[kSCSeedEnergy] = MEData("SCSeedEnergy", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);

    yaxis.nbins = 50;
    yaxis.low = 0.;
    yaxis.high = 150.;
    _data[kSCClusterVsSeed] = MEData("SCClusterVsSeed", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH2F, &xaxis, &yaxis);

    _data[kSCSeedOccupancy] = MEData("SCSeedOccupancy", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);
    _data[kSingleCrystalCluster] = MEData("SCSingleCrystalCluster", BinService::kEcal3P, BinService::kSuperCrystal, MonitorElement::DQM_KIND_TH2F);

    xaxis.nbins = 20;
    xaxis.low = 0.;
    xaxis.high = 20.;
    _data[kSCNum] = MEData("SCNum", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);

    xaxis.nbins = 15;
    xaxis.low = 0.;
    xaxis.high = 15.;
    _data[kSCNBCs] = MEData("SCNBCs", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);
    xaxis.nbins = 50;
    xaxis.low = 0.;
    xaxis.high = 150.;
    _data[kSCNcrystals] = MEData("SCNcrystals", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);

    xaxis.nbins = 50;
    xaxis.low = 0.;
    xaxis.high = 1.2;
    _data[kSCR9] = MEData("SCR9", BinService::kEcal2P, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);


    xaxis.nbins = 50;
    xaxis.low = 0.;
    xaxis.high = 0.5;
    _data[kPi0] = MEData("Pi0", BinService::nObjType, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);
    xaxis.low = 2.9;
    xaxis.high = 3.3;
    _data[kJPsi] = MEData("JPsi", BinService::nObjType, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);
    xaxis.low = 40.;
    xaxis.high = 110.;
    _data[kZ] = MEData("Z", BinService::nObjType, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);
    xaxis.low = 110.;
    xaxis.high = 3000.;
    _data[kHighMass] = MEData("HighMass", BinService::nObjType, BinService::kUser, MonitorElement::DQM_KIND_TH1F, &xaxis);
  }

  DEFINE_ECALDQM_WORKER(ClusterTask);
}

