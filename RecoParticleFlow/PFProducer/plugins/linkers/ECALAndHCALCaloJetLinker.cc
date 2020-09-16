#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

class ECALAndHCALCaloJetLinker : public BlockElementLinkerBase {
public:
  ECALAndHCALCaloJetLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

  double testLink(size_t ielem1,
                  size_t ielem2,
                  reco::PFBlockElement::Type type1,
                  reco::PFBlockElement::Type type2,
                  const ElementListConst& elements,
                  const PFTables& tables,
                  const reco::PFMultiLinksIndex& multilinks) const override;

private:
  bool useKDTree_, debug_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, ECALAndHCALCaloJetLinker, "ECALAndHCALCaloJetLinker");

double ECALAndHCALCaloJetLinker::testLink(size_t ielem1,
                                          size_t ielem2,
                                          reco::PFBlockElement::Type type1,
                                          reco::PFBlockElement::Type type2,
                                          const ElementListConst& elements,
                                          const PFTables& tables,
                                          const reco::PFMultiLinksIndex& multilinks) const {
  const auto* elem1 = elements[ielem1];
  const auto* elem2 = elements[ielem2];
  const reco::PFBlockElementCluster *hcalelem(nullptr), *ecalelem(nullptr);
  double dist(-1.0);
  if (type1 < type2) {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    ecalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFClusterRef& ecalref = ecalelem->clusterRef();
  const reco::PFClusterRef& hcalref = hcalelem->clusterRef();
  const reco::PFCluster::REPPoint& ecalreppos = ecalref->positionREP();
  if (hcalref.isNull() || ecalref.isNull()) {
    throw cms::Exception("BadClusterRefs") << "PFBlockElementCluster's refs are null!";
  }
  //dist = ( std::abs(ecalreppos.Eta()) > 2.5 ?
  //	   LinkByRecHit::computeDist( ecalreppos.Eta(),
  //				      ecalreppos.Phi(),
  // 				      hcalref->positionREP().Eta(),
  //				      hcalref->positionREP().Phi() )
  //	   : -1.0 );
  // return (dist < 0.2 ? dist : -1.0);
  dist = LinkByRecHit::computeDist(
      ecalreppos.Eta(), ecalreppos.Phi(), hcalref->positionREP().Eta(), hcalref->positionREP().Phi());

  return (dist < 0.2 ? dist : -1.0);
}
