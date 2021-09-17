#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonSelectorAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockElementSCEqual.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
// for single tower H/E
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"

//quick pT for superclusters
inline double ptFast(const double energy, const math::XYZPoint& position, const math::XYZPoint& origin) {
  const auto v = position - origin;
  return energy * std::sqrt(v.perp2() / v.mag2());
}

#include <memory>

#include <unordered_map>

class SuperClusterImporter : public BlockElementImporterBase {
public:
  SuperClusterImporter(const edm::ParameterSet&, edm::ConsumesCollector&);

  void updateEventSetup(const edm::EventSetup& es) override;

  void importToBlock(const edm::Event&, ElementList&) const override;

private:
  edm::EDGetTokenT<reco::SuperClusterCollection> _srcEB, _srcEE;
  edm::EDGetTokenT<CaloTowerCollection> _srcTowers;
  const double _maxHoverE, _pTbyPass, _minSCPt;
  CaloTowerConstituentsMap const* towerMap_;
  bool _superClustersArePF;
  static const math::XYZPoint _zero;

  const edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> _ctmapToken;
};

const math::XYZPoint SuperClusterImporter::_zero = math::XYZPoint(0, 0, 0);

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, SuperClusterImporter, "SuperClusterImporter");

SuperClusterImporter::SuperClusterImporter(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
    : BlockElementImporterBase(conf, cc),
      _srcEB(cc.consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("source_eb"))),
      _srcEE(cc.consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("source_ee"))),
      _srcTowers(cc.consumes<CaloTowerCollection>(conf.getParameter<edm::InputTag>("source_towers"))),
      _maxHoverE(conf.getParameter<double>("maximumHoverE")),
      _pTbyPass(conf.getParameter<double>("minPTforBypass")),
      _minSCPt(conf.getParameter<double>("minSuperClusterPt")),
      _superClustersArePF(conf.getParameter<bool>("superClustersArePF")),
      _ctmapToken(cc.esConsumes<edm::Transition::BeginLuminosityBlock>()) {}

void SuperClusterImporter::updateEventSetup(const edm::EventSetup& es) { towerMap_ = &es.getData(_ctmapToken); }

void SuperClusterImporter::importToBlock(const edm::Event& e, BlockElementImporterBase::ElementList& elems) const {
  auto eb_scs = e.getHandle(_srcEB);
  auto ee_scs = e.getHandle(_srcEE);
  auto const& towers = e.get(_srcTowers);
  elems.reserve(elems.size() + eb_scs->size() + ee_scs->size());
  // setup our elements so that all the SCs are grouped together
  auto SCs_end =
      std::partition(elems.begin(), elems.end(), [](auto const& a) { return a->type() == reco::PFBlockElement::SC; });
  // add eb superclusters
  auto bsc = eb_scs->cbegin();
  auto esc = eb_scs->cend();
  reco::PFBlockElementSuperCluster* scbe = nullptr;
  reco::SuperClusterRef scref;
  for (auto sc = bsc; sc != esc; ++sc) {
    scref = reco::SuperClusterRef(eb_scs, std::distance(bsc, sc));
    PFBlockElementSCEqual myEqual(scref);
    auto sc_elem = std::find_if(elems.begin(), SCs_end, myEqual);
    const double scpT = ptFast(sc->energy(), sc->position(), _zero);
    const auto towersBehindCluster = egamma::towersOf(*sc, *towerMap_);
    const double H_tower =
        (egamma::depth1HcalESum(towersBehindCluster, towers) + egamma::depth2HcalESum(towersBehindCluster, towers));
    const double HoverE = H_tower / sc->energy();
    if (sc_elem == SCs_end && scpT > _minSCPt && (scpT > _pTbyPass || HoverE < _maxHoverE)) {
      scbe = new reco::PFBlockElementSuperCluster(scref);
      scbe->setFromPFSuperCluster(_superClustersArePF);
      SCs_end = elems.emplace(SCs_end, scbe);
      ++SCs_end;  // point to element *after* the new one
    }
  }  // loop on eb superclusters
  // add ee superclusters
  bsc = ee_scs->cbegin();
  esc = ee_scs->cend();
  for (auto sc = bsc; sc != esc; ++sc) {
    scref = reco::SuperClusterRef(ee_scs, std::distance(bsc, sc));
    PFBlockElementSCEqual myEqual(scref);
    auto sc_elem = std::find_if(elems.begin(), SCs_end, myEqual);
    const double scpT = ptFast(sc->energy(), sc->position(), _zero);
    const auto towersBehindCluster = egamma::towersOf(*sc, *towerMap_);
    const double H_tower =
        (egamma::depth1HcalESum(towersBehindCluster, towers) + egamma::depth2HcalESum(towersBehindCluster, towers));
    const double HoverE = H_tower / sc->energy();
    if (sc_elem == SCs_end && scpT > _minSCPt && (scpT > _pTbyPass || HoverE < _maxHoverE)) {
      scbe = new reco::PFBlockElementSuperCluster(scref);
      scbe->setFromPFSuperCluster(_superClustersArePF);
      SCs_end = elems.emplace(SCs_end, scbe);
      ++SCs_end;  // point to element *after* the new one
    }
  }  // loop on ee superclusters
  elems.shrink_to_fit();
}
