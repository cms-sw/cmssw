#include "FWPFBlockProxyBuilder.h"
#include "Fireworks/Candidates/interface/FWLegoCandidate.h"

//______________________________________________________________________________
void FWPFBlockProxyBuilder::scaleProduct(TEveElementList *parent, FWViewType::EType viewType, const FWViewContext *vc) {
  typedef std::vector<ScalableLines> Lines_t;
  FWViewEnergyScale *caloScale = vc->getEnergyScale();

  if (viewType == FWViewType::kRhoPhiPF ||
      viewType == FWViewType::kRhoZ) { /* Handle the rhophi and rhoz cluster scaling */
    for (Lines_t::iterator i = m_clusters.begin(); i != m_clusters.end(); ++i) {
      if (vc == (*i).m_vc) {
        float value = caloScale->getPlotEt() ? (*i).m_et : (*i).m_energy;
        (*i).m_ls->SetScale(caloScale->getScaleFactor3D() * value);
        TEveProjected *proj = *(*i).m_ls->BeginProjecteds();
        proj->UpdateProjection();
      }
    }
  } /* Handle cluster scaling in lego view(s) */
  else if (viewType == FWViewType::kLego || viewType == FWViewType::kLegoPFECAL) {  // Loop products
    for (TEveElement::List_i i = parent->BeginChildren(); i != parent->EndChildren(); ++i) {
      if ((*i)->HasChildren()) {  // Loop elements of block
        for (TEveElement::List_i j = (*i)->BeginChildren(); j != (*i)->EndChildren(); ++j) {
          if (strcmp((*j)->GetElementName(), "BlockCluster") == 0) {
            FWLegoCandidate *cluster = dynamic_cast<FWLegoCandidate *>(*j);
            cluster->updateScale(vc, context());
          }
        }
      }
    }
  }
}

//______________________________________________________________________________
void FWPFBlockProxyBuilder::setupTrackElement(const reco::PFBlockElement &blockElement,
                                              TEveElement &oItemHolder,
                                              const FWViewContext *vc,
                                              FWViewType::EType viewType) {
  if (blockElement.trackType(reco::PFBlockElement::DEFAULT)) {
    reco::Track track = *blockElement.trackRef();
    FWPFTrackUtils *trackUtils = new FWPFTrackUtils();

    if (viewType == FWViewType::kLego || viewType == FWViewType::kLegoPFECAL)  // Lego views
    {
      TEveStraightLineSet *legoTrack = trackUtils->setupLegoTrack(track);
      setupAddElement(legoTrack, &oItemHolder);
    } else if (viewType == FWViewType::kRhoPhiPF)  // RhoPhi view
    {
      TEveTrack *trk = trackUtils->setupTrack(track);
      TEvePointSet *ps = trackUtils->getCollisionMarkers(trk);
      setupAddElement(trk, &oItemHolder);
      if (ps->GetN() != 0)
        setupAddElement(ps, &oItemHolder);
      else
        delete ps;
    } else if (viewType == FWViewType::kRhoZ)  // RhoZ view
    {
      TEveTrack *trk = trackUtils->setupTrack(track);
      TEvePointSet *markers = trackUtils->getCollisionMarkers(trk);
      TEvePointSet *ps = new TEvePointSet();
      setupAddElement(trk, &oItemHolder);

      Float_t *trackPoints = trk->GetP();
      unsigned int last = (trk->GetN() - 1) * 3;
      float y = trackPoints[last + 1];
      float z = trackPoints[last + 2];

      // Reposition any points that have been translated in RhoZ
      for (signed int i = 0; i < markers->GetN(); ++i) {
        Float_t a, b, c;
        markers->GetPoint(i, a, b, c);

        if (y < 0 && b > 0)
          b *= -1;
        else if (y > 0 && b < 0)
          b *= -1;

        if (z < 0 && c > 0)
          c *= -1;
        else if (z > 0 && c < 0)
          c *= -1;

        ps->SetNextPoint(a, b, c);
      }

      if (ps->GetN() != 0)
        setupAddElement(ps, &oItemHolder);
      else
        delete ps;
      delete markers;
    }

    delete trackUtils;
  }
}

//______________________________________________________________________________
void FWPFBlockProxyBuilder::setupClusterElement(const reco::PFBlockElement &blockElement,
                                                TEveElement &oItemHolder,
                                                const FWViewContext *vc,
                                                FWViewType::EType viewType,
                                                float r) {
  // Get reference to PFCluster
  reco::PFCluster cluster = *blockElement.clusterRef();
  TEveVector centre = TEveVector(cluster.x(), cluster.y(), cluster.z());
  float energy = cluster.energy();
  float et = FWPFMaths::calculateEt(centre, energy);
  float pt = et;
  float eta = cluster.eta();
  float phi = cluster.phi();

  FWProxyBuilderBase::context().voteMaxEtAndEnergy(et, energy);

  if (viewType == FWViewType::kLego || viewType == FWViewType::kLegoPFECAL) {
    FWLegoCandidate *legoCluster = new FWLegoCandidate(vc, FWProxyBuilderBase::context(), energy, et, pt, eta, phi);
    legoCluster->SetMarkerColor(FWProxyBuilderBase::item()->defaultDisplayProperties().color());
    legoCluster->SetElementName("BlockCluster");
    setupAddElement(legoCluster, &oItemHolder);
  }
  if (viewType == FWViewType::kRhoPhiPF) {
    const FWDisplayProperties &dp = item()->defaultDisplayProperties();
    FWPFClusterRPZUtils *clusterUtils = new FWPFClusterRPZUtils();
    TEveScalableStraightLineSet *rpCluster = clusterUtils->buildRhoPhiClusterLineSet(cluster, vc, energy, et, r);
    rpCluster->SetLineColor(dp.color());
    m_clusters.push_back(ScalableLines(rpCluster, et, energy, vc));
    setupAddElement(rpCluster, &oItemHolder);
    delete clusterUtils;
  } else if (viewType == FWViewType::kRhoZ) {
    const FWDisplayProperties &dp = item()->defaultDisplayProperties();
    FWPFClusterRPZUtils *clusterUtils = new FWPFClusterRPZUtils();
    TEveScalableStraightLineSet *rzCluster = clusterUtils->buildRhoZClusterLineSet(
        cluster, vc, context().caloTransAngle(), energy, et, r, context().caloZ1());
    rzCluster->SetLineColor(dp.color());
    m_clusters.push_back(ScalableLines(rzCluster, et, energy, vc));
    setupAddElement(rzCluster, &oItemHolder);
    delete clusterUtils;
  }
}

//______________________________________________________________________________
void FWPFBlockProxyBuilder::buildViewType(const reco::PFBlock &iData,
                                          unsigned int iIndex,
                                          TEveElement &oItemHolder,
                                          FWViewType::EType viewType,
                                          const FWViewContext *vc) {
  const edm::OwnVector<reco::PFBlockElement> &elements = iData.elements();

  for (unsigned int i = 0; i < elements.size(); ++i) {
    reco::PFBlockElement::Type type = elements[i].type();
    switch (type) {
      case 1:  // TRACK
        if (e_builderType == BASE)
          setupTrackElement(elements[i], oItemHolder, vc, viewType);
        break;

      case 4:  // ECAL
        if (e_builderType == ECAL)
          setupClusterElement(elements[i], oItemHolder, vc, viewType, FWPFGeom::caloR1());
        break;

      case 5:  // HCAL
        if (e_builderType == HCAL) {
          if (viewType == FWViewType::kRhoPhiPF)
            setupClusterElement(elements[i], oItemHolder, vc, viewType, FWPFGeom::caloR2());
          else  // RhoZ
            setupClusterElement(elements[i], oItemHolder, vc, viewType, context().caloR1());
        }
        break;

      default:  // Ignore anything that isn't wanted
        break;
    }
  }
}

//______________________________________________________________________________
REGISTER_FWPROXYBUILDER(FWPFBlockProxyBuilder,
                        reco::PFBlock,
                        "PF Block",
                        FWViewType::kRhoPhiPFBit | FWViewType::kLegoBit | FWViewType::kRhoZBit |
                            FWViewType::kLegoPFECALBit);
REGISTER_FWPROXYBUILDER(FWPFBlockEcalProxyBuilder,
                        reco::PFBlock,
                        "PF Block",
                        FWViewType::kLegoPFECALBit | FWViewType::kRhoPhiPFBit | FWViewType::kRhoZBit);
REGISTER_FWPROXYBUILDER(FWPFBlockHcalProxyBuilder,
                        reco::PFBlock,
                        "PF Block",
                        FWViewType::kLegoBit | FWViewType::kRhoPhiPFBit | FWViewType::kRhoZBit);
