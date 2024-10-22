#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonSelectorAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockElementSCEqual.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

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
  const double _maxHoverE, _pTbyPass, _minSCPt;
  const edm::EDGetTokenT<HBHERecHitCollection> hbheRecHitsTag_;
  const int maxSeverityHB_;
  const int maxSeverityHE_;
  bool cutsFromDB;
  CaloTowerConstituentsMap const* towerMap_;
  CaloGeometry const* caloGeom_;
  HcalTopology const* hcalTopo_;
  HcalChannelQuality const* hcalChannelQual_;
  HcalSeverityLevelComputer const* hcalSev_;
  bool _superClustersArePF;
  static const math::XYZPoint _zero;

  const edm::ESGetToken<CaloTowerConstituentsMap, CaloGeometryRecord> _ctmapToken;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
  const edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalTopologyToken_;
  const edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> hcalChannelQualityToken_;
  const edm::ESGetToken<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd> hcalSevLvlComputerToken_;
  edm::ESGetToken<HcalPFCuts, HcalPFCutsRcd> hcalCutsToken_;
  HcalPFCuts const* hcalCuts = nullptr;
};

const math::XYZPoint SuperClusterImporter::_zero = math::XYZPoint(0, 0, 0);

DEFINE_EDM_PLUGIN(BlockElementImporterFactory, SuperClusterImporter, "SuperClusterImporter");

SuperClusterImporter::SuperClusterImporter(const edm::ParameterSet& conf, edm::ConsumesCollector& cc)
    : BlockElementImporterBase(conf, cc),
      _srcEB(cc.consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("source_eb"))),
      _srcEE(cc.consumes<reco::SuperClusterCollection>(conf.getParameter<edm::InputTag>("source_ee"))),
      _maxHoverE(conf.getParameter<double>("maximumHoverE")),
      _pTbyPass(conf.getParameter<double>("minPTforBypass")),
      _minSCPt(conf.getParameter<double>("minSuperClusterPt")),
      hbheRecHitsTag_(cc.consumes(conf.getParameter<edm::InputTag>("hbheRecHitsTag"))),
      maxSeverityHB_(conf.getParameter<int>("maxSeverityHB")),
      maxSeverityHE_(conf.getParameter<int>("maxSeverityHE")),
      cutsFromDB(conf.getParameter<bool>("usePFThresholdsFromDB")),
      _superClustersArePF(conf.getParameter<bool>("superClustersArePF")),
      _ctmapToken(cc.esConsumes<edm::Transition::BeginLuminosityBlock>()),
      caloGeometryToken_{cc.esConsumes<edm::Transition::BeginLuminosityBlock>()},
      hcalTopologyToken_{cc.esConsumes<edm::Transition::BeginLuminosityBlock>()},
      hcalChannelQualityToken_{cc.esConsumes<edm::Transition::BeginLuminosityBlock>(edm::ESInputTag("", "withTopo"))},
      hcalSevLvlComputerToken_{cc.esConsumes<edm::Transition::BeginLuminosityBlock>()} {
  if (cutsFromDB) {
    hcalCutsToken_ = cc.esConsumes<HcalPFCuts, HcalPFCutsRcd, edm::Transition::BeginLuminosityBlock>(
        edm::ESInputTag("", "withTopo"));
  }
}

void SuperClusterImporter::updateEventSetup(const edm::EventSetup& es) {
  towerMap_ = &es.getData(_ctmapToken);
  if (cutsFromDB) {
    hcalCuts = &es.getData(hcalCutsToken_);
  }
  caloGeom_ = &es.getData(caloGeometryToken_);
  hcalTopo_ = &es.getData(hcalTopologyToken_);
  hcalChannelQual_ = &es.getData(hcalChannelQualityToken_);
  hcalSev_ = &es.getData(hcalSevLvlComputerToken_);
}

void SuperClusterImporter::importToBlock(const edm::Event& e, BlockElementImporterBase::ElementList& elems) const {
  auto eb_scs = e.getHandle(_srcEB);
  auto ee_scs = e.getHandle(_srcEE);
  elems.reserve(elems.size() + eb_scs->size() + ee_scs->size());
  // setup our elements so that all the SCs are grouped together
  auto SCs_end =
      std::partition(elems.begin(), elems.end(), [](auto const& a) { return a->type() == reco::PFBlockElement::SC; });
  // add eb superclusters
  auto bsc = eb_scs->cbegin();
  auto esc = eb_scs->cend();
  reco::PFBlockElementSuperCluster* scbe = nullptr;
  reco::SuperClusterRef scref;

  EgammaHcalIsolation thisHcalVar_ = EgammaHcalIsolation(EgammaHcalIsolation::InclusionRule::isBehindClusterSeed,
                                                         0,  //outercone
                                                         EgammaHcalIsolation::InclusionRule::withinConeAroundCluster,
                                                         0,  //innercone
                                                         {{0, 0, 0, 0}},
                                                         {{0, 0, 0, 0}},
                                                         maxSeverityHB_,
                                                         {{0, 0, 0, 0, 0, 0, 0}},
                                                         {{0, 0, 0, 0, 0, 0, 0}},
                                                         maxSeverityHE_,
                                                         e.get(hbheRecHitsTag_),
                                                         caloGeom_,
                                                         hcalTopo_,
                                                         hcalChannelQual_,
                                                         hcalSev_,
                                                         towerMap_);

  for (auto sc = bsc; sc != esc; ++sc) {
    scref = reco::SuperClusterRef(eb_scs, std::distance(bsc, sc));
    PFBlockElementSCEqual myEqual(scref);
    auto sc_elem = std::find_if(elems.begin(), SCs_end, myEqual);
    const double scpT = ptFast(sc->energy(), sc->position(), _zero);
    const double H_tower = thisHcalVar_.getHcalESumBc(scref.get(), 0, hcalCuts);
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
    const double H_tower = thisHcalVar_.getHcalESumBc(scref.get(), 0, hcalCuts);
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
