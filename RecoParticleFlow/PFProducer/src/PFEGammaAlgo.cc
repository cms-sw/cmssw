#include "RecoParticleFlow/PFProducer/interface/PFEGammaAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <TFile.h>
#include <TVector2.h>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <numeric>
#include <TMath.h>

// include combinations header (was never incorporated in boost)
#include "combination.hpp"

// just for now do this
//#define PFLOW_DEBUG

#ifdef PFLOW_DEBUG
#define docast(x, y) dynamic_cast<x>(y)
#define LOGVERB(x) edm::LogVerbatim(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) edm::LogInfo(x)
#else
#define docast(x, y) reinterpret_cast<x>(y)
#define LOGVERB(x) LogTrace(x)
#define LOGWARN(x) edm::LogWarning(x)
#define LOGERR(x) edm::LogError(x)
#define LOGDRESSED(x) LogDebug(x)
#endif

using namespace std;
using namespace reco;
using namespace std::placeholders;  // for _1, _2, _3...

namespace {
  typedef PFEGammaAlgo::PFSCElement SCElement;
  typedef PFEGammaAlgo::EEtoPSAssociation EEtoPSAssociation;
  typedef std::pair<CaloClusterPtr::key_type, CaloClusterPtr> EEtoPSElement;
  typedef PFEGammaAlgo::PFClusterElement ClusterElement;

  class SeedMatchesToProtoObject {
  public:
    SeedMatchesToProtoObject(const reco::ElectronSeedRef& s)
        : scfromseed_(s->caloCluster().castTo<reco::SuperClusterRef>()) {
      ispfsc_ = false;
      if (scfromseed_.isNonnull()) {
        const edm::Ptr<reco::PFCluster> testCast(scfromseed_->seed());
        ispfsc_ = testCast.isNonnull();
      }
    }
    bool operator()(const PFEGammaAlgo::ProtoEGObject& po) {
      if (scfromseed_.isNull() || !po.parentSC)
        return false;
      if (ispfsc_) {
        return (scfromseed_->seed() == po.parentSC->superClusterRef()->seed());
      }
      return (scfromseed_->seed()->seed() == po.parentSC->superClusterRef()->seed()->seed());
    }

  private:
    const reco::SuperClusterRef scfromseed_;
    bool ispfsc_;
  };

  template <bool useConvs = false>
  bool elementNotCloserToOther(const reco::PFBlockRef& block,
                               const PFBlockElement::Type& keytype,
                               const size_t key,
                               const PFBlockElement::Type& valtype,
                               const size_t test,
                               const float EoPin_cut = 1.0e6) {
    constexpr reco::PFBlockElement::TrackType ConvType = reco::PFBlockElement::T_FROM_GAMMACONV;
    // this is inside out but I just want something that works right now
    switch (keytype) {
      case reco::PFBlockElement::GSF: {
        const reco::PFBlockElementGsfTrack* elemasgsf =
            docast(const reco::PFBlockElementGsfTrack*, &(block->elements()[key]));
        if (elemasgsf && valtype == PFBlockElement::ECAL) {
          const ClusterElement* elemasclus = reinterpret_cast<const ClusterElement*>(&(block->elements()[test]));
          float cluster_e = elemasclus->clusterRef()->correctedEnergy();
          float trk_pin = elemasgsf->Pin().P();
          if (cluster_e / trk_pin > EoPin_cut) {
            LOGDRESSED("elementNotCloserToOther") << "GSF track failed EoP cut to match with cluster!";
            return false;
          }
        }
      } break;
      case reco::PFBlockElement::TRACK: {
        const reco::PFBlockElementTrack* elemaskf = docast(const reco::PFBlockElementTrack*, &(block->elements()[key]));
        if (elemaskf && valtype == PFBlockElement::ECAL) {
          const ClusterElement* elemasclus = reinterpret_cast<const ClusterElement*>(&(block->elements()[test]));
          float cluster_e = elemasclus->clusterRef()->correctedEnergy();
          float trk_pin = std::sqrt(elemaskf->trackRef()->innerMomentum().mag2());
          if (cluster_e / trk_pin > EoPin_cut) {
            LOGDRESSED("elementNotCloserToOther") << "KF track failed EoP cut to match with cluster!";
            return false;
          }
        }
      } break;
      default:
        break;
    }

    const float dist = block->dist(key, test, block->linkData(), reco::PFBlock::LINKTEST_ALL);
    if (dist == -1.0f)
      return false;  // don't associate non-linked elems
    std::multimap<double, unsigned> dists_to_val;
    block->associatedElements(test, block->linkData(), dists_to_val, keytype, reco::PFBlock::LINKTEST_ALL);

    for (const auto& valdist : dists_to_val) {
      const size_t idx = valdist.second;
      // check track types for conversion info
      switch (keytype) {
        case reco::PFBlockElement::GSF: {
          const reco::PFBlockElementGsfTrack* elemasgsf =
              docast(const reco::PFBlockElementGsfTrack*, &(block->elements()[idx]));
          if (!useConvs && elemasgsf->trackType(ConvType))
            return false;
          if (elemasgsf && valtype == PFBlockElement::ECAL) {
            const ClusterElement* elemasclus = docast(const ClusterElement*, &(block->elements()[test]));
            float cluster_e = elemasclus->clusterRef()->correctedEnergy();
            float trk_pin = elemasgsf->Pin().P();
            if (cluster_e / trk_pin > EoPin_cut)
              continue;
          }
        } break;
        case reco::PFBlockElement::TRACK: {
          const reco::PFBlockElementTrack* elemaskf =
              docast(const reco::PFBlockElementTrack*, &(block->elements()[idx]));
          if (!useConvs && elemaskf->trackType(ConvType))
            return false;
          if (elemaskf && valtype == PFBlockElement::ECAL) {
            const ClusterElement* elemasclus = reinterpret_cast<const ClusterElement*>(&(block->elements()[test]));
            float cluster_e = elemasclus->clusterRef()->correctedEnergy();
            float trk_pin = std::sqrt(elemaskf->trackRef()->innerMomentum().mag2());
            if (cluster_e / trk_pin > EoPin_cut)
              continue;
          }
        } break;
        default:
          break;
      }
      if (valdist.first < dist && idx != key) {
        LOGDRESSED("elementNotCloserToOther")
            << "key element of type " << keytype << " is closer to another element of type" << valtype << std::endl;
        return false;  // false if closer element of specified type found
      }
    }
    return true;
  }

  template <class Element1, class Element2>
  bool compatibleEoPOut(const Element1& e, const Element2& comp) {
    if (PFBlockElement::ECAL != e.type()) {
      return false;
    }
    const ClusterElement& elemascluster = docast(ClusterElement const&, e);
    const float gsf_eta_diff = std::abs(comp.positionAtECALEntrance().eta() - comp.Pout().eta());
    const reco::PFClusterRef& cRef = elemascluster.clusterRef();
    return (gsf_eta_diff <= 0.3 && cRef->energy() / comp.Pout().t() <= 5);
  }

  constexpr reco::PFBlockElement::TrackType ConvType = reco::PFBlockElement::T_FROM_GAMMACONV;

  template <PFBlockElement::Type keytype, PFBlockElement::Type valtype, bool useConv = false>

  struct NotCloserToOther {
    const reco::PFBlockElement* comp;
    const reco::PFBlockRef& block;
    const float EoPin_cut;
    NotCloserToOther(const reco::PFBlockRef& b, const reco::PFBlockElement* e, const float EoPcut = 1.0e6)
        : comp(e), block(b), EoPin_cut(EoPcut) {}
    template <class T>
    bool operator()(const T& e) {
      if (!e.flag() || valtype != e->type())
        return false;
      return elementNotCloserToOther<useConv>(block, keytype, comp->index(), valtype, e->index(), EoPin_cut);
    }
  };

  struct LesserByDistance {
    const reco::PFBlockElement* comp;
    const reco::PFBlockRef& block;
    const reco::PFBlock::LinkData& links;
    LesserByDistance(const reco::PFBlockRef& b, const reco::PFBlock::LinkData& l, const reco::PFBlockElement* e)
        : comp(e), block(b), links(l) {}
    bool operator()(FlaggedPtr<const reco::PFBlockElement> const& e1,
                    FlaggedPtr<const reco::PFBlockElement> const& e2) {
      double dist1 = block->dist(comp->index(), e1->index(), links, reco::PFBlock::LINKTEST_ALL);
      double dist2 = block->dist(comp->index(), e2->index(), links, reco::PFBlock::LINKTEST_ALL);
      dist1 = (dist1 == -1.0 ? 1e6 : dist1);
      dist2 = (dist2 == -1.0 ? 1e6 : dist2);
      return dist1 < dist2;
    }
  };

  bool isROLinkedByClusterOrTrack(const PFEGammaAlgo::ProtoEGObject& RO1, const PFEGammaAlgo::ProtoEGObject& RO2) {
    // also don't allow ROs where both have clusters
    // and GSF tracks to merge (10 Dec 2013)
    if (!RO1.primaryGSFs.empty() && !RO2.primaryGSFs.empty()) {
      LOGDRESSED("isROLinkedByClusterOrTrack") << "cannot merge, both have GSFs!" << std::endl;
      return false;
    }
    // don't allow EB/EE to mix (11 Sept 2013)
    if (!RO1.ecalclusters.empty() && !RO2.ecalclusters.empty()) {
      if (RO1.ecalclusters.front()->clusterRef()->layer() != RO2.ecalclusters.front()->clusterRef()->layer()) {
        LOGDRESSED("isROLinkedByClusterOrTrack") << "cannot merge, different ECAL types!" << std::endl;
        return false;
      }
    }
    const reco::PFBlockRef& blk = RO1.parentBlock;
    bool not_closer;
    // check links track -> cluster
    for (const auto& cluster : RO1.ecalclusters) {
      for (const auto& primgsf : RO2.primaryGSFs) {
        not_closer = elementNotCloserToOther(blk, cluster->type(), cluster->index(), primgsf->type(), primgsf->index());
        if (not_closer) {
          LOGDRESSED("isROLinkedByClusterOrTrack") << "merged by cluster to primary GSF" << std::endl;
          return true;
        } else {
          LOGDRESSED("isROLinkedByClusterOrTrack") << "cluster to primary GSF failed since"
                                                   << " cluster closer to another GSF" << std::endl;
        }
      }
      for (const auto& primkf : RO2.primaryKFs) {
        not_closer = elementNotCloserToOther(blk, cluster->type(), cluster->index(), primkf->type(), primkf->index());
        if (not_closer) {
          LOGDRESSED("isROLinkedByClusterOrTrack") << "merged by cluster to primary KF" << std::endl;
          return true;
        }
      }
      for (const auto& secdkf : RO2.secondaryKFs) {
        not_closer = elementNotCloserToOther(blk, cluster->type(), cluster->index(), secdkf->type(), secdkf->index());
        if (not_closer) {
          LOGDRESSED("isROLinkedByClusterOrTrack") << "merged by cluster to secondary KF" << std::endl;
          return true;
        }
      }
      // check links brem -> cluster
      for (const auto& brem : RO2.brems) {
        not_closer = elementNotCloserToOther(blk, cluster->type(), cluster->index(), brem->type(), brem->index());
        if (not_closer) {
          LOGDRESSED("isROLinkedByClusterOrTrack") << "merged by cluster to brem KF" << std::endl;
          return true;
        }
      }
    }
    // check links primary gsf -> secondary kf
    for (const auto& primgsf : RO1.primaryGSFs) {
      for (const auto& secdkf : RO2.secondaryKFs) {
        not_closer = elementNotCloserToOther(blk, primgsf->type(), primgsf->index(), secdkf->type(), secdkf->index());
        if (not_closer) {
          LOGDRESSED("isROLinkedByClusterOrTrack") << "merged by GSF to secondary KF" << std::endl;
          return true;
        }
      }
    }
    // check links primary kf -> secondary kf
    for (const auto& primkf : RO1.primaryKFs) {
      for (const auto& secdkf : RO2.secondaryKFs) {
        not_closer = elementNotCloserToOther(blk, primkf->type(), primkf->index(), secdkf->type(), secdkf->index());
        if (not_closer) {
          LOGDRESSED("isROLinkedByClusterOrTrack") << "merged by primary KF to secondary KF" << std::endl;
          return true;
        }
      }
    }
    // check links secondary kf -> secondary kf
    for (const auto& secdkf1 : RO1.secondaryKFs) {
      for (const auto& secdkf2 : RO2.secondaryKFs) {
        not_closer =
            elementNotCloserToOther<true>(blk, secdkf1->type(), secdkf1->index(), secdkf2->type(), secdkf2->index());
        if (not_closer) {
          LOGDRESSED("isROLinkedByClusterOrTrack") << "merged by secondary KF to secondary KF" << std::endl;
          return true;
        }
      }
    }
    return false;
  }

  bool testIfROMergableByLink(const PFEGammaAlgo::ProtoEGObject& ro, PFEGammaAlgo::ProtoEGObject& comp) {
    const bool result = (isROLinkedByClusterOrTrack(comp, ro) || isROLinkedByClusterOrTrack(ro, comp));
    return result;
  }

  std::vector<const ClusterElement*> getSCAssociatedECALsSafe(
      const reco::SuperClusterRef& scref, std::vector<FlaggedPtr<const reco::PFBlockElement>>& ecals) {
    std::vector<const ClusterElement*> cluster_list;
    auto sccl = scref->clustersBegin();
    auto scend = scref->clustersEnd();
    auto pfc = ecals.begin();
    auto pfcend = ecals.end();
    for (; sccl != scend; ++sccl) {
      std::vector<const ClusterElement*> matched_pfcs;
      const double eg_energy = (*sccl)->energy();

      for (pfc = ecals.begin(); pfc != pfcend; ++pfc) {
        const ClusterElement* pfcel = docast(const ClusterElement*, pfc->get());
        const bool matched = ClusterClusterMapping::overlap(**sccl, *(pfcel->clusterRef()));
        // need to protect against high energy clusters being attached
        // to low-energy SCs
        if (matched && pfcel->clusterRef()->energy() < 1.2 * scref->energy()) {
          matched_pfcs.push_back(pfcel);
        }
      }
      std::sort(matched_pfcs.begin(), matched_pfcs.end());

      double min_residual = 1e6;
      std::vector<const ClusterElement*> best_comb;
      for (size_t i = 1; i <= matched_pfcs.size(); ++i) {
        //now we find the combination of PF-clusters which
        //has the smallest energy residual with respect to the
        //EG-cluster we are looking at now
        do {
          double energy = std::accumulate(
              matched_pfcs.begin(), matched_pfcs.begin() + i, 0.0, [](const double a, const ClusterElement* c) {
                return a + c->clusterRef()->energy();
              });
          const double resid = std::abs(energy - eg_energy);
          if (resid < min_residual) {
            best_comb.clear();
            best_comb.reserve(i);
            min_residual = resid;
            best_comb.insert(best_comb.begin(), matched_pfcs.begin(), matched_pfcs.begin() + i);
          }
        } while (notboost::next_combination(matched_pfcs.begin(), matched_pfcs.begin() + i, matched_pfcs.end()));
      }
      for (const auto& clelem : best_comb) {
        if (std::find(cluster_list.begin(), cluster_list.end(), clelem) == cluster_list.end()) {
          cluster_list.push_back(clelem);
        }
      }
    }
    return cluster_list;
  }
  bool addPFClusterToROSafe(const ClusterElement* cl, PFEGammaAlgo::ProtoEGObject& RO) {
    if (RO.ecalclusters.empty()) {
      RO.ecalclusters.emplace_back(cl, true);
      return true;
    } else {
      const PFLayer::Layer clayer = cl->clusterRef()->layer();
      const PFLayer::Layer blayer = RO.ecalclusters.back()->clusterRef()->layer();
      if (clayer == blayer) {
        RO.ecalclusters.emplace_back(cl, true);
        return true;
      }
    }
    return false;
  }

  // sets the cluster best associated to the GSF track
  // leave it null if no GSF track
  void setROElectronCluster(PFEGammaAlgo::ProtoEGObject& RO) {
    if (RO.ecalclusters.empty())
      return;
    RO.lateBrem = -1;
    RO.firstBrem = -1;
    RO.nBremsWithClusters = -1;
    const reco::PFBlockElementBrem *firstBrem = nullptr, *lastBrem = nullptr;
    const reco::PFBlockElementCluster *bremCluster = nullptr, *gsfCluster = nullptr, *kfCluster = nullptr,
                                      *gsfCluster_noassc = nullptr;
    const reco::PFBlockRef& parent = RO.parentBlock;
    int nBremClusters = 0;
    constexpr float maxDist = 1e6;
    float mDist_gsf(maxDist), mDist_gsf_noassc(maxDist), mDist_kf(maxDist);
    for (const auto& cluster : RO.ecalclusters) {
      for (const auto& gsf : RO.primaryGSFs) {
        const bool hasclu =
            elementNotCloserToOther(parent, gsf->type(), gsf->index(), cluster->type(), cluster->index());
        const float deta = std::abs(cluster->clusterRef()->positionREP().eta() - gsf->positionAtECALEntrance().eta());
        const float dphi = std::abs(
            TVector2::Phi_mpi_pi(cluster->clusterRef()->positionREP().phi() - gsf->positionAtECALEntrance().phi()));
        const float dist = std::hypot(deta, dphi);
        if (hasclu && dist < mDist_gsf) {
          gsfCluster = cluster.get();
          mDist_gsf = dist;
        } else if (dist < mDist_gsf_noassc) {
          gsfCluster_noassc = cluster.get();
          mDist_gsf_noassc = dist;
        }
      }
      for (const auto& kf : RO.primaryKFs) {
        const bool hasclu = elementNotCloserToOther(parent, kf->type(), kf->index(), cluster->type(), cluster->index());
        const float dist = parent->dist(cluster->index(), kf->index(), parent->linkData(), reco::PFBlock::LINKTEST_ALL);
        if (hasclu && dist < mDist_kf) {
          kfCluster = cluster.get();
          mDist_kf = dist;
        }
      }
      for (const auto& brem : RO.brems) {
        const bool hasclu =
            elementNotCloserToOther(parent, brem->type(), brem->index(), cluster->type(), cluster->index());
        if (hasclu) {
          ++nBremClusters;
          if (!firstBrem || (firstBrem->indTrajPoint() - 2 > brem->indTrajPoint() - 2)) {
            firstBrem = brem;
          }
          if (!lastBrem || (lastBrem->indTrajPoint() - 2 < brem->indTrajPoint() - 2)) {
            lastBrem = brem;
            bremCluster = cluster.get();
          }
        }
      }
    }
    if (!gsfCluster && !kfCluster && !bremCluster) {
      gsfCluster = gsfCluster_noassc;
    }
    RO.nBremsWithClusters = nBremClusters;
    RO.lateBrem = 0;
    if (gsfCluster) {
      RO.electronClusters.push_back(gsfCluster);
    } else if (kfCluster) {
      RO.electronClusters.push_back(kfCluster);
    }
    if (bremCluster && !gsfCluster && !kfCluster) {
      RO.electronClusters.push_back(bremCluster);
    }
    if (firstBrem && RO.ecalclusters.size() > 1) {
      RO.firstBrem = firstBrem->indTrajPoint() - 2;
      if (bremCluster == gsfCluster)
        RO.lateBrem = 1;
    }
  }
}  // namespace

PFEGammaAlgo::PFEGammaAlgo(const PFEGammaAlgo::PFEGConfigInfo& cfg,
                           GBRForests const& gbrForests,
                           EEtoPSAssociation const& eetops,
                           ESEEIntercalibConstants const& esEEInterCalib,
                           ESChannelStatus const& channelStatus,
                           reco::Vertex const& primaryVertex)
    : gbrForests_(gbrForests),
      eetops_(eetops),
      cfg_(cfg),
      primaryVertex_(primaryVertex),
      channelStatus_(channelStatus) {
  thePFEnergyCalibration_.initAlphaGamma_ESplanes_fromDB(&esEEInterCalib);
}

float PFEGammaAlgo::evaluateSingleLegMVA(const reco::PFBlockRef& blockRef,
                                         const reco::Vertex& primaryVtx,
                                         unsigned int trackIndex) {
  const reco::PFBlock& block = *blockRef;
  const edm::OwnVector<reco::PFBlockElement>& elements = block.elements();
  //use this to store linkdata in the associatedElements function below
  const PFBlock::LinkData& linkData = block.linkData();
  //calculate MVA Variables
  const float chi2 = elements[trackIndex].trackRef()->chi2() / elements[trackIndex].trackRef()->ndof();
  const float nlost = elements[trackIndex].trackRef()->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS);
  const float nLayers = elements[trackIndex].trackRef()->hitPattern().trackerLayersWithMeasurement();
  const float trackPt = elements[trackIndex].trackRef()->pt();
  const float stip = elements[trackIndex].trackRefPF()->STIP();

  float linkedE = 0;
  float linkedH = 0;
  std::multimap<double, unsigned int> ecalAssoTrack;
  block.associatedElements(
      trackIndex, linkData, ecalAssoTrack, reco::PFBlockElement::ECAL, reco::PFBlock::LINKTEST_ALL);
  std::multimap<double, unsigned int> hcalAssoTrack;
  block.associatedElements(
      trackIndex, linkData, hcalAssoTrack, reco::PFBlockElement::HCAL, reco::PFBlock::LINKTEST_ALL);
  if (!ecalAssoTrack.empty()) {
    for (auto& itecal : ecalAssoTrack) {
      linkedE = linkedE + elements[itecal.second].clusterRef()->energy();
    }
  }
  if (!hcalAssoTrack.empty()) {
    for (auto& ithcal : hcalAssoTrack) {
      linkedH = linkedH + elements[ithcal.second].clusterRef()->energy();
    }
  }
  const float eOverPt = linkedE / elements[trackIndex].trackRef()->pt();
  const float hOverPt = linkedH / elements[trackIndex].trackRef()->pt();
  GlobalVector rvtx(elements[trackIndex].trackRef()->innerPosition().X() - primaryVtx.x(),
                    elements[trackIndex].trackRef()->innerPosition().Y() - primaryVtx.y(),
                    elements[trackIndex].trackRef()->innerPosition().Z() - primaryVtx.z());
  double vtxPhi = rvtx.phi();
  //delta Phi between conversion vertex and track
  float delPhi = fabs(deltaPhi(vtxPhi, elements[trackIndex].trackRef()->innerMomentum().Phi()));

  float vars[] = {delPhi, nLayers, chi2, eOverPt, hOverPt, trackPt, stip, nlost};

  return gbrForests_.singleLeg_->GetAdaBoostClassifier(vars);
}

bool PFEGammaAlgo::isMuon(const reco::PFBlockElement& pfbe) {
  switch (pfbe.type()) {
    case reco::PFBlockElement::GSF: {
      auto& elements = _currentblock->elements();
      std::multimap<double, unsigned> tks;
      _currentblock->associatedElements(
          pfbe.index(), _currentlinks, tks, reco::PFBlockElement::TRACK, reco::PFBlock::LINKTEST_ALL);
      for (const auto& tk : tks) {
        if (PFMuonAlgo::isMuon(elements[tk.second])) {
          return true;
        }
      }
    } break;
    case reco::PFBlockElement::TRACK:
      return PFMuonAlgo::isMuon(pfbe);
      break;
    default:
      break;
  }
  return false;
}

PFEGammaAlgo::EgammaObjects PFEGammaAlgo::operator()(const reco::PFBlockRef& block) {
  LOGVERB("PFEGammaAlgo") << "Resetting PFEGammaAlgo for new block and running!" << std::endl;

  // candidate collections:
  // this starts off as an inclusive list of prototype objects built from
  // supercluster/ecal-driven seeds and tracker driven seeds in a block
  // it is then refined through by various cleanings, determining the energy
  // flow.
  // use list for constant-time removals
  std::list<ProtoEGObject> refinableObjects;

  _splayedblock.clear();
  _splayedblock.resize(13);  // make sure that we always have the HGCAL entry

  _currentblock = block;
  _currentlinks = block->linkData();
  //LOGDRESSED("PFEGammaAlgo") << *_currentblock << std::endl;
  LOGVERB("PFEGammaAlgo") << "Splaying block" << std::endl;
  //unwrap the PF block into a fast access map
  for (const auto& pfelement : _currentblock->elements()) {
    if (isMuon(pfelement))
      continue;  // don't allow muons in our element list
    if (pfelement.type() == PFBlockElement::HCAL && pfelement.clusterRef()->flags() & reco::CaloCluster::badHcalMarker)
      continue;  // skip also dead area markers for now
    const size_t itype = (size_t)pfelement.type();
    if (itype >= _splayedblock.size())
      _splayedblock.resize(itype + 1);
    _splayedblock[itype].emplace_back(&pfelement, true);
  }

  // show the result of splaying the tree if it's really *really* needed
#ifdef PFLOW_DEBUG
  std::stringstream splayout;
  for (size_t itype = 0; itype < _splayedblock.size(); ++itype) {
    splayout << "\tType: " << itype << " indices: ";
    for (const auto& flaggedelement : _splayedblock[itype]) {
      splayout << flaggedelement->index() << ' ';
    }
    if (itype != _splayedblock.size() - 1)
      splayout << std::endl;
  }
  LOGVERB("PFEGammaAlgo") << splayout.str();
#endif

  // precleaning of the ECAL clusters with respect to primary KF tracks
  // we don't allow clusters in super clusters to be locked out this way
  removeOrLinkECALClustersToKFTracks();

  initializeProtoCands(refinableObjects);
  LOGDRESSED("PFEGammaAlgo") << "Initialized " << refinableObjects.size() << " proto-EGamma objects" << std::endl;
  dumpCurrentRefinableObjects();

  //
  // now we start the refining steps
  //
  //

  // --- Primary Linking Step ---
  // since this is particle flow and we try to work from the pixels out
  // we start by linking the tracks together and finding the ECAL clusters
  for (auto& RO : refinableObjects) {
    // find the KF tracks associated to GSF primary tracks
    linkRefinableObjectGSFTracksToKFs(RO);
    // do the same for HCAL clusters associated to the GSF
    linkRefinableObjectPrimaryGSFTrackToHCAL(RO);
    // link secondary KF tracks associated to primary KF tracks
    linkRefinableObjectPrimaryKFsToSecondaryKFs(RO);
    // pick up clusters that are linked to the GSF primary
    linkRefinableObjectPrimaryGSFTrackToECAL(RO);
    // link associated KF to ECAL (ECAL part grabs PS clusters too if able)
    linkRefinableObjectKFTracksToECAL(RO);
    // now finally look for clusters associated to brem tangents
    linkRefinableObjectBremTangentsToECAL(RO);
  }

  LOGDRESSED("PFEGammaAlgo") << "Dumping after GSF and KF Track (Primary) Linking : " << std::endl;
  dumpCurrentRefinableObjects();

  // merge objects after primary linking
  mergeROsByAnyLink(refinableObjects);

  LOGDRESSED("PFEGammaAlgo") << "Dumping after first merging operation : " << std::endl;
  dumpCurrentRefinableObjects();

  // --- Secondary Linking Step ---
  // after this we go through the ECAL clusters on the remaining tracks
  // and try to link those in...
  for (auto& RO : refinableObjects) {
    // look for conversion legs
    linkRefinableObjectECALToSingleLegConv(RO);
    dumpCurrentRefinableObjects();
    // look for tracks that complement conversion legs
    linkRefinableObjectConvSecondaryKFsToSecondaryKFs(RO);
    // look again for ECAL clusters (this time with an e/p cut)
    linkRefinableObjectSecondaryKFsToECAL(RO);
  }

  LOGDRESSED("PFEGammaAlgo") << "Dumping after ECAL to Track (Secondary) Linking : " << std::endl;
  dumpCurrentRefinableObjects();

  // merge objects after primary linking
  mergeROsByAnyLink(refinableObjects);

  LOGDRESSED("PFEGammaAlgo") << "There are " << refinableObjects.size() << " after the 2nd merging step." << std::endl;
  dumpCurrentRefinableObjects();

  // -- unlinking and proto-object vetos, final sorting
  for (auto& RO : refinableObjects) {
    // remove secondary KFs (and possibly ECALs) matched to HCAL clusters
    unlinkRefinableObjectKFandECALMatchedToHCAL(RO, false, false);
    // remove secondary KFs and ECALs linked to them that have bad E/p_in
    // and spoil the resolution
    unlinkRefinableObjectKFandECALWithBadEoverP(RO);
    // put things back in order after partitioning
    std::sort(RO.ecalclusters.begin(), RO.ecalclusters.end(), [](auto const& a, auto const& b) {
      return (a->clusterRef()->correctedEnergy() > b->clusterRef()->correctedEnergy());
    });
    setROElectronCluster(RO);
  }

  LOGDRESSED("PFEGammaAlgo") << "There are " << refinableObjects.size() << " after the unlinking and vetos step."
                             << std::endl;
  dumpCurrentRefinableObjects();

  // fill the PF candidates and then build the refined SC
  return fillPFCandidates(refinableObjects);
}

void PFEGammaAlgo::initializeProtoCands(std::list<PFEGammaAlgo::ProtoEGObject>& egobjs) {
  // step 1: build SC based proto-candidates
  // in the future there will be an SC Et requirement made here to control
  // block size
  for (auto& element : _splayedblock[PFBlockElement::SC]) {
    LOGDRESSED("PFEGammaAlgo") << "creating SC-based proto-object" << std::endl
                               << "\tSC at index: " << element->index() << " has type: " << element->type()
                               << std::endl;
    element.setFlag(false);
    ProtoEGObject fromSC;
    fromSC.nBremsWithClusters = -1;
    fromSC.firstBrem = -1;
    fromSC.lateBrem = -1;
    fromSC.parentBlock = _currentblock;
    fromSC.parentSC = docast(const PFSCElement*, element.get());
    // splay the supercluster so we can knock out used elements
    bool sc_success = unwrapSuperCluster(fromSC.parentSC, fromSC.ecalclusters, fromSC.ecal2ps);
    if (sc_success) {
      /*
      auto ins_pos = std::lower_bound(refinableObjects.begin(),
				      refinableObjects.end(),
				      fromSC,
				      [&](const ProtoEGObject& a,
					  const ProtoEGObject& b){
					const double a_en = 
					a.parentSC->superClusterRef()->energy();
					const double b_en = 
					b.parentSC->superClusterRef()->energy();
					return a_en < b_en;
				      });
      */
      egobjs.insert(egobjs.end(), fromSC);
    }
  }
  // step 2: build GSF-seed-based proto-candidates
  reco::GsfTrackRef gsfref_forextra;
  reco::TrackExtraRef gsftrk_extra;
  reco::ElectronSeedRef theseedref;
  for (auto& element : _splayedblock[PFBlockElement::GSF]) {
    LOGDRESSED("PFEGammaAlgo") << "creating GSF-based proto-object" << std::endl
                               << "\tGSF at index: " << element->index() << " has type: " << element->type()
                               << std::endl;
    const PFGSFElement* elementAsGSF = docast(const PFGSFElement*, element.get());
    if (elementAsGSF->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)) {
      continue;  // for now, do not allow dedicated brems to make proto-objects
    }
    element.setFlag(false);

    ProtoEGObject fromGSF;
    fromGSF.nBremsWithClusters = -1;
    fromGSF.firstBrem = -1;
    fromGSF.lateBrem = 0;
    gsfref_forextra = elementAsGSF->GsftrackRef();
    gsftrk_extra = (gsfref_forextra.isAvailable() ? gsfref_forextra->extra() : reco::TrackExtraRef());
    theseedref = (gsftrk_extra.isAvailable() ? gsftrk_extra->seedRef().castTo<reco::ElectronSeedRef>()
                                             : reco::ElectronSeedRef());
    fromGSF.electronSeed = theseedref;
    // exception if there's no seed
    if (fromGSF.electronSeed.isNull() || !fromGSF.electronSeed.isAvailable()) {
      std::stringstream gsf_err;
      elementAsGSF->Dump(gsf_err, "\t");
      throw cms::Exception("PFEGammaAlgo::initializeProtoCands()")
          << "Found a GSF track with no seed! This should not happen!" << std::endl
          << gsf_err.str() << std::endl;
    }
    // flag this GSF element as globally used and push back the track ref
    // into the protocand
    element.setFlag(false);
    fromGSF.parentBlock = _currentblock;
    fromGSF.primaryGSFs.push_back(elementAsGSF);
    // add the directly matched brem tangents
    for (auto& brem : _splayedblock[PFBlockElement::BREM]) {
      float dist =
          _currentblock->dist(elementAsGSF->index(), brem->index(), _currentlinks, reco::PFBlock::LINKTEST_ALL);
      if (dist == 0.001f) {
        const PFBremElement* eAsBrem = docast(const PFBremElement*, brem.get());
        fromGSF.brems.push_back(eAsBrem);
        fromGSF.localMap.insert(eAsBrem, elementAsGSF);
        brem.setFlag(false);
      }
    }
    // if this track is ECAL seeded reset links or import cluster
    // tracker (this is pixel only, right?) driven seeds just get the GSF
    // track associated since this only branches for ECAL Driven seeds
    if (fromGSF.electronSeed->isEcalDriven()) {
      // step 2a: either merge with existing ProtoEG object with SC or add
      //          SC directly to this proto EG object if not present
      LOGDRESSED("PFEGammaAlgo") << "GSF-based proto-object is ECAL driven, merging SC-cand" << std::endl;
      LOGVERB("PFEGammaAlgo") << "ECAL Seed Ptr: " << fromGSF.electronSeed.get()
                              << " isAvailable: " << fromGSF.electronSeed.isAvailable()
                              << " isNonnull: " << fromGSF.electronSeed.isNonnull() << std::endl;
      SeedMatchesToProtoObject sctoseedmatch(fromGSF.electronSeed);
      auto objsbegin = egobjs.begin();
      auto objsend = egobjs.end();
      // this auto is a std::list<ProtoEGObject>::iterator
      auto clusmatch = std::find_if(objsbegin, objsend, sctoseedmatch);
      if (clusmatch != objsend) {
        fromGSF.parentSC = clusmatch->parentSC;
        fromGSF.ecalclusters = std::move(clusmatch->ecalclusters);
        fromGSF.ecal2ps = std::move(clusmatch->ecal2ps);
        egobjs.erase(clusmatch);
      } else if (fromGSF.electronSeed.isAvailable() && fromGSF.electronSeed.isNonnull()) {
        // link tests in the gap region can current split a gap electron
        // HEY THIS IS A WORK AROUND FOR A KNOWN BUG IN PFBLOCKALGO
        // MAYBE WE SHOULD FIX IT??????????????????????????????????
        LOGDRESSED("PFEGammaAlgo") << "Encountered the known GSF-SC splitting bug "
                                   << " in PFBlockAlgo! We should really fix this!" << std::endl;
      } else {  // SC was not in a earlier proto-object
        std::stringstream gsf_err;
        elementAsGSF->Dump(gsf_err, "\t");
        throw cms::Exception("PFEGammaAlgo::initializeProtoCands()")
            << "Expected SuperCluster from ECAL driven GSF seed "
            << "was not found in the block!" << std::endl
            << gsf_err.str() << std::endl;
      }  // supercluster in block
    }    // is ECAL driven seed?

    egobjs.insert(egobjs.end(), fromGSF);
  }  // end loop on GSF elements of block
}

bool PFEGammaAlgo::unwrapSuperCluster(const PFSCElement* thesc,
                                      std::vector<FlaggedPtr<const PFClusterElement>>& ecalclusters,
                                      ClusterMap& ecal2ps) {
  ecalclusters.clear();
  ecal2ps.clear();
  LOGVERB("PFEGammaAlgo") << "Pointer to SC element: 0x" << std::hex << thesc << std::dec << std::endl
                          << "cleared ecalclusters and ecal2ps!" << std::endl;
  auto ecalbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ecalend = _splayedblock[reco::PFBlockElement::ECAL].end();
  auto hgcalbegin = _splayedblock[reco::PFBlockElement::HGCAL].begin();
  auto hgcalend = _splayedblock[reco::PFBlockElement::HGCAL].end();
  if (ecalbegin == ecalend && hgcalbegin == hgcalend) {
    LOGERR("PFEGammaAlgo::unwrapSuperCluster()") << "There are no ECAL elements in a block with imported SC!"
                                                 << " This is a bug we should fix this!" << std::endl;
    return false;
  }
  reco::SuperClusterRef scref = thesc->superClusterRef();
  const bool is_pf_sc = thesc->fromPFSuperCluster();
  if (!(scref.isAvailable() && scref.isNonnull())) {
    throw cms::Exception("PFEGammaAlgo::unwrapSuperCluster()")
        << "SuperCluster pointed to by block element is null!" << std::endl;
  }
  LOGDRESSED("PFEGammaAlgo") << "Got a valid super cluster ref! 0x" << std::hex << scref.get() << std::dec << std::endl;
  const size_t nscclusters = scref->clustersSize();
  const size_t nscpsclusters = scref->preshowerClustersSize();
  size_t npfpsclusters = 0;
  size_t npfclusters = 0;
  LOGDRESSED("PFEGammaAlgo") << "Precalculated cluster multiplicities: " << nscclusters << ' ' << nscpsclusters
                             << std::endl;
  NotCloserToOther<reco::PFBlockElement::SC, reco::PFBlockElement::ECAL> ecalClustersInSC(_currentblock, thesc);
  NotCloserToOther<reco::PFBlockElement::SC, reco::PFBlockElement::HGCAL> hgcalClustersInSC(_currentblock, thesc);
  auto ecalfirstnotinsc = std::partition(ecalbegin, ecalend, ecalClustersInSC);
  auto hgcalfirstnotinsc = std::partition(hgcalbegin, hgcalend, hgcalClustersInSC);
  //reset the begin and end iterators
  ecalbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  ecalend = _splayedblock[reco::PFBlockElement::ECAL].end();

  hgcalbegin = _splayedblock[reco::PFBlockElement::HGCAL].begin();
  hgcalend = _splayedblock[reco::PFBlockElement::HGCAL].end();

  //get list of associated clusters by det id and energy matching
  //(only needed when using non-pf supercluster)
  std::vector<const ClusterElement*> safePFClusters =
      is_pf_sc ? std::vector<const ClusterElement*>()
               : getSCAssociatedECALsSafe(scref, _splayedblock[reco::PFBlockElement::ECAL]);

  if (ecalfirstnotinsc == ecalbegin && hgcalfirstnotinsc == hgcalbegin) {
    LOGERR("PFEGammaAlgo::unwrapSuperCluster()") << "No associated block elements to SuperCluster!"
                                                 << " This is a bug we should fix!" << std::endl;
    return false;
  }
  npfclusters = std::distance(ecalbegin, ecalfirstnotinsc) + std::distance(hgcalbegin, hgcalfirstnotinsc);
  // ensure we have found the correct number of PF ecal clusters in the case
  // that this is a PF supercluster, otherwise all bets are off
  if (is_pf_sc && nscclusters != npfclusters) {
    std::stringstream sc_err;
    thesc->Dump(sc_err, "\t");
    throw cms::Exception("PFEGammaAlgo::unwrapSuperCluster()")
        << "The number of found ecal elements (" << nscclusters << ") in block is not the same as"
        << " the number of ecal PF clusters reported by the PFSuperCluster"
        << " itself (" << npfclusters << ")! This should not happen!" << std::endl
        << sc_err.str() << std::endl;
  }
  for (auto ecalitr = ecalbegin; ecalitr != ecalfirstnotinsc; ++ecalitr) {
    const PFClusterElement* elemascluster = docast(const PFClusterElement*, ecalitr->get());

    // reject clusters that really shouldn't be associated to the SC
    // (only needed when using non-pf-supercluster)
    if (!is_pf_sc && std::find(safePFClusters.begin(), safePFClusters.end(), elemascluster) == safePFClusters.end())
      continue;

    //add cluster
    ecalclusters.emplace_back(elemascluster, true);
    //mark cluster as used
    ecalitr->setFlag(false);

    // process the ES elements
    // auto is a pair<Iterator,bool> here, bool is false when placing fails
    auto emplaceresult = ecal2ps.emplace(elemascluster, ClusterMap::mapped_type());
    if (!emplaceresult.second) {
      std::stringstream clus_err;
      elemascluster->Dump(clus_err, "\t");
      throw cms::Exception("PFEGammaAlgo::unwrapSuperCluster()")
          << "List of pointers to ECAL block elements contains non-unique items!"
          << " This is very bad!" << std::endl
          << "cluster ptr = 0x" << std::hex << elemascluster << std::dec << std::endl
          << clus_err.str() << std::endl;
    }
    ClusterMap::mapped_type& eslist = emplaceresult.first->second;
    npfpsclusters += attachPSClusters(elemascluster, eslist);
  }  // loop over ecal elements

  for (auto hgcalitr = hgcalbegin; hgcalitr != hgcalfirstnotinsc; ++hgcalitr) {
    const PFClusterElement* elemascluster = docast(const PFClusterElement*, hgcalitr->get());

    // reject clusters that really shouldn't be associated to the SC
    // (only needed when using non-pf-supercluster)
    if (!is_pf_sc && std::find(safePFClusters.begin(), safePFClusters.end(), elemascluster) == safePFClusters.end())
      continue;

    //add cluster
    ecalclusters.emplace_back(elemascluster, true);
    //mark cluster as used
    hgcalitr->setFlag(false);
  }  // loop over ecal elements

  /*
   if( is_pf_sc && nscpsclusters != npfpsclusters) {
     std::stringstream sc_err;
     thesc->Dump(sc_err,"\t");
     throw cms::Exception("PFEGammaAlgo::unwrapSuperCluster()")
       << "The number of found PF preshower elements (" 
       << npfpsclusters << ") in block is not the same as"
       << " the number of preshower clusters reported by the PFSuperCluster"
       << " itself (" << nscpsclusters << ")! This should not happen!" 
       << std::endl 
       << sc_err.str() << std::endl;
   }
   */

  LOGDRESSED("PFEGammaAlgo") << " Unwrapped SC has " << npfclusters << " ECAL sub-clusters"
                             << " and " << npfpsclusters << " PreShower layers 1 & 2 clusters!" << std::endl;
  return true;
}

int PFEGammaAlgo::attachPSClusters(const ClusterElement* ecalclus, ClusterMap::mapped_type& eslist) {
  if (ecalclus->clusterRef()->layer() == PFLayer::ECAL_BARREL)
    return 0;
  edm::Ptr<reco::PFCluster> clusptr = refToPtr(ecalclus->clusterRef());
  EEtoPSElement ecalkey(clusptr.key(), clusptr);
  auto assc_ps =
      std::equal_range(eetops_.cbegin(), eetops_.cend(), ecalkey, [](const EEtoPSElement& a, const EEtoPSElement& b) {
        return a.first < b.first;
      });
  for (const auto& ps1 : _splayedblock[reco::PFBlockElement::PS1]) {
    edm::Ptr<reco::PFCluster> temp = refToPtr(ps1->clusterRef());
    for (auto pscl = assc_ps.first; pscl != assc_ps.second; ++pscl) {
      if (pscl->second == temp) {
        const ClusterElement* pstemp = docast(const ClusterElement*, ps1.get());
        eslist.emplace_back(pstemp);
      }
    }
  }
  for (const auto& ps2 : _splayedblock[reco::PFBlockElement::PS2]) {
    edm::Ptr<reco::PFCluster> temp = refToPtr(ps2->clusterRef());
    for (auto pscl = assc_ps.first; pscl != assc_ps.second; ++pscl) {
      if (pscl->second == temp) {
        const ClusterElement* pstemp = docast(const ClusterElement*, ps2.get());
        eslist.emplace_back(pstemp);
      }
    }
  }
  return eslist.size();
}

void PFEGammaAlgo::dumpCurrentRefinableObjects() const {
#ifdef PFLOW_DEBUG
  edm::LogVerbatim("PFEGammaAlgo")
      //<< "Dumping current block: " << std::endl << *_currentblock << std::endl
      << "Dumping " << refinableObjects.size() << " refinable objects for this block: " << std::endl;
  for (const auto& ro : refinableObjects) {
    std::stringstream info;
    info << "Refinable Object:" << std::endl;
    if (ro.parentSC) {
      info << "\tSuperCluster element attached to object:" << std::endl << '\t';
      ro.parentSC->Dump(info, "\t");
      info << std::endl;
    }
    if (ro.electronSeed.isNonnull()) {
      info << "\tGSF element attached to object:" << std::endl;
      ro.primaryGSFs.front()->Dump(info, "\t");
      info << std::endl;
      info << "firstBrem : " << ro.firstBrem << " lateBrem : " << ro.lateBrem
           << " nBrems with cluster : " << ro.nBremsWithClusters << std::endl;
      ;
      if (ro.electronClusters.size() && ro.electronClusters[0]) {
        info << "electron cluster : ";
        ro.electronClusters[0]->Dump(info, "\t");
        info << std::endl;
      } else {
        info << " no electron cluster." << std::endl;
      }
    }
    if (ro.primaryKFs.size()) {
      info << "\tPrimary KF tracks attached to object: " << std::endl;
      for (const auto& kf : ro.primaryKFs) {
        kf->Dump(info, "\t");
        info << std::endl;
      }
    }
    if (ro.secondaryKFs.size()) {
      info << "\tSecondary KF tracks attached to object: " << std::endl;
      for (const auto& kf : ro.secondaryKFs) {
        kf->Dump(info, "\t");
        info << std::endl;
      }
    }
    if (ro.brems.size()) {
      info << "\tBrem tangents attached to object: " << std::endl;
      for (const auto& brem : ro.brems) {
        brem->Dump(info, "\t");
        info << std::endl;
      }
    }
    if (ro.ecalclusters.size()) {
      info << "\tECAL clusters attached to object: " << std::endl;
      for (const auto& clus : ro.ecalclusters) {
        clus->Dump(info, "\t");
        info << std::endl;
        if (ro.ecal2ps.find(clus) != ro.ecal2ps.end()) {
          for (const auto& psclus : ro.ecal2ps.at(clus)) {
            info << "\t\t Attached PS Cluster: ";
            psclus->Dump(info, "");
            info << std::endl;
          }
        }
      }
    }
    edm::LogVerbatim("PFEGammaAlgo") << info.str();
  }
#endif
}

// look through our KF tracks in this block and match
void PFEGammaAlgo::removeOrLinkECALClustersToKFTracks() {
  typedef std::multimap<double, unsigned> MatchedMap;
  typedef const reco::PFBlockElementGsfTrack* GsfTrackElementPtr;
  if (_splayedblock[reco::PFBlockElement::ECAL].empty() || _splayedblock[reco::PFBlockElement::TRACK].empty())
    return;
  MatchedMap matchedGSFs, matchedECALs;
  std::unordered_map<GsfTrackElementPtr, MatchedMap> gsf_ecal_cache;
  for (auto& kftrack : _splayedblock[reco::PFBlockElement::TRACK]) {
    matchedGSFs.clear();
    _currentblock->associatedElements(
        kftrack->index(), _currentlinks, matchedGSFs, reco::PFBlockElement::GSF, reco::PFBlock::LINKTEST_ALL);
    if (matchedGSFs.empty()) {  // only run this if we aren't associated to GSF
      LesserByDistance closestTrackToECAL(_currentblock, _currentlinks, kftrack.get());
      auto ecalbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
      auto ecalend = _splayedblock[reco::PFBlockElement::ECAL].end();
      std::partial_sort(ecalbegin, ecalbegin + 1, ecalend, closestTrackToECAL);
      auto& closestECAL = _splayedblock[reco::PFBlockElement::ECAL].front();
      const float dist =
          _currentblock->dist(kftrack->index(), closestECAL->index(), _currentlinks, reco::PFBlock::LINKTEST_ALL);
      bool inSC = false;
      for (auto& sc : _splayedblock[reco::PFBlockElement::SC]) {
        float dist_sc =
            _currentblock->dist(sc->index(), closestECAL->index(), _currentlinks, reco::PFBlock::LINKTEST_ALL);
        if (dist_sc != -1.0f) {
          inSC = true;
          break;
        }
      }

      if (dist != -1.0f && closestECAL.flag()) {
        bool gsflinked = false;
        // check that this cluster is not associated to a GSF track
        for (const auto& gsfflag : _splayedblock[reco::PFBlockElement::GSF]) {
          const reco::PFBlockElementGsfTrack* elemasgsf = docast(const reco::PFBlockElementGsfTrack*, gsfflag.get());
          if (elemasgsf->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)) {
            continue;  // keep clusters that have a found conversion GSF near
          }
          // make sure cache exists
          if (!gsf_ecal_cache.count(elemasgsf)) {
            matchedECALs.clear();
            _currentblock->associatedElements(elemasgsf->index(),
                                              _currentlinks,
                                              matchedECALs,
                                              reco::PFBlockElement::ECAL,
                                              reco::PFBlock::LINKTEST_ALL);
            gsf_ecal_cache.emplace(elemasgsf, matchedECALs);
            MatchedMap().swap(matchedECALs);
          }
          const MatchedMap& ecal_matches = gsf_ecal_cache[elemasgsf];
          if (!ecal_matches.empty()) {
            if (ecal_matches.begin()->second == closestECAL->index()) {
              gsflinked = true;
              break;
            }
          }
        }  // loop over primary GSF tracks
        if (!gsflinked && !inSC) {
          // determine if we should remove the matched cluster
          const reco::PFBlockElementTrack* kfEle = docast(const reco::PFBlockElementTrack*, kftrack.get());
          const reco::TrackRef& trackref = kfEle->trackRef();

          const int nexhits = trackref->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS);
          bool fromprimaryvertex = false;
          for (auto vtxtks = primaryVertex_.tracks_begin(); vtxtks != primaryVertex_.tracks_end(); ++vtxtks) {
            if (trackref == vtxtks->castTo<reco::TrackRef>()) {
              fromprimaryvertex = true;
              break;
            }
          }  // loop over tracks in primary vertex
             // if associated to good non-GSF matched track remove this cluster
          if (PFTrackAlgoTools::isGoodForEGMPrimary(trackref->algo()) && nexhits == 0 && fromprimaryvertex) {
            closestECAL.setFlag(false);
          }
        }
      }  // found a good closest ECAL match
    }    // no GSF track matched to KF
  }      // loop over KF elements
}

void PFEGammaAlgo::mergeROsByAnyLink(std::list<PFEGammaAlgo::ProtoEGObject>& ROs) {
  if (ROs.size() < 2)
    return;  // nothing to do with one or zero ROs
  bool check_for_merge = true;
  while (check_for_merge) {
    // bugfix for early termination merging loop (15 April 2014)
    // check all pairwise combinations in the list
    // if one has a merge shuffle it to the front of the list
    // if there are no merges left to do we can terminate
    for (auto it1 = ROs.begin(); it1 != ROs.end(); ++it1) {
      auto find_start = it1;
      ++find_start;
      auto has_merge = std::find_if(find_start, ROs.end(), std::bind(testIfROMergableByLink, _1, *it1));
      if (has_merge != ROs.end() && it1 != ROs.begin()) {
        std::swap(*(ROs.begin()), *it1);
        break;
      }
    }  // ensure mergables are shuffled to the front
    ProtoEGObject& thefront = ROs.front();
    auto mergestart = ROs.begin();
    ++mergestart;
    auto nomerge = std::partition(mergestart, ROs.end(), std::bind(testIfROMergableByLink, _1, thefront));
    if (nomerge != mergestart) {
      LOGDRESSED("PFEGammaAlgo::mergeROsByAnyLink()")
          << "Found objects " << std::distance(mergestart, nomerge) << " to merge by links to the front!" << std::endl;
      for (auto roToMerge = mergestart; roToMerge != nomerge; ++roToMerge) {
        //bugfix! L.Gray 14 Jan 2016
        // -- check that the front is still mergeable!
        if (!thefront.ecalclusters.empty() && !roToMerge->ecalclusters.empty()) {
          if (thefront.ecalclusters.front()->clusterRef()->layer() !=
              roToMerge->ecalclusters.front()->clusterRef()->layer()) {
            LOGWARN("PFEGammaAlgo::mergeROsByAnyLink") << "Tried to merge EB and EE clusters! Skipping!";
            ROs.push_back(*roToMerge);
            continue;
          }
        }
        //end bugfix
        thefront.ecalclusters.insert(
            thefront.ecalclusters.end(), roToMerge->ecalclusters.begin(), roToMerge->ecalclusters.end());
        thefront.ecal2ps.insert(roToMerge->ecal2ps.begin(), roToMerge->ecal2ps.end());
        thefront.secondaryKFs.insert(
            thefront.secondaryKFs.end(), roToMerge->secondaryKFs.begin(), roToMerge->secondaryKFs.end());

        thefront.localMap.concatenate(roToMerge->localMap);
        // TO FIX -> use best (E_gsf - E_clustersum)/E_GSF
        if (!thefront.parentSC && roToMerge->parentSC) {
          thefront.parentSC = roToMerge->parentSC;
        }
        if (thefront.electronSeed.isNull() && roToMerge->electronSeed.isNonnull()) {
          thefront.electronSeed = roToMerge->electronSeed;
          thefront.primaryGSFs.insert(
              thefront.primaryGSFs.end(), roToMerge->primaryGSFs.begin(), roToMerge->primaryGSFs.end());
          thefront.primaryKFs.insert(
              thefront.primaryKFs.end(), roToMerge->primaryKFs.begin(), roToMerge->primaryKFs.end());
          thefront.brems.insert(thefront.brems.end(), roToMerge->brems.begin(), roToMerge->brems.end());
          thefront.electronClusters = roToMerge->electronClusters;
          thefront.nBremsWithClusters = roToMerge->nBremsWithClusters;
          thefront.firstBrem = roToMerge->firstBrem;
          thefront.lateBrem = roToMerge->lateBrem;
        } else if (thefront.electronSeed.isNonnull() && roToMerge->electronSeed.isNonnull()) {
          LOGDRESSED("PFEGammaAlgo::mergeROsByAnyLink")
              << "Need to implement proper merging of two gsf candidates!" << std::endl;
        }
      }
      ROs.erase(mergestart, nomerge);
      // put the merged element in the back of the cleaned list
      ROs.push_back(ROs.front());
      ROs.pop_front();
    } else {
      check_for_merge = false;
    }
  }
  LOGDRESSED("PFEGammaAlgo::mergeROsByAnyLink()")
      << "After merging by links there are: " << ROs.size() << " refinable EGamma objects!" << std::endl;
}

// pull in KF tracks associated to the RO but not closer to another
// NB: in initializeProtoCands() we forced the GSF tracks not to be
//     from a conversion, but we will leave a protection here just in
//     case things change in the future
void PFEGammaAlgo::linkRefinableObjectGSFTracksToKFs(ProtoEGObject& RO) {
  constexpr reco::PFBlockElement::TrackType convType = reco::PFBlockElement::T_FROM_GAMMACONV;
  if (_splayedblock[reco::PFBlockElement::TRACK].empty())
    return;
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend = _splayedblock[reco::PFBlockElement::TRACK].end();
  for (auto& gsfflagged : RO.primaryGSFs) {
    const PFGSFElement* seedtk = gsfflagged;
    // don't process SC-only ROs or secondary seeded ROs
    if (RO.electronSeed.isNull() || seedtk->trackType(convType))
      continue;
    NotCloserToOther<reco::PFBlockElement::GSF, reco::PFBlockElement::TRACK> gsfTrackToKFs(_currentblock, seedtk);
    // get KF tracks not closer to another and not already used
    auto notlinked = std::partition(KFbegin, KFend, gsfTrackToKFs);
    // attach tracks and set as used
    for (auto kft = KFbegin; kft != notlinked; ++kft) {
      const PFKFElement* elemaskf = docast(const PFKFElement*, kft->get());
      // don't care about things that aren't primaries or directly
      // associated secondary tracks
      if (isPrimaryTrack(*elemaskf, *seedtk) && !elemaskf->trackType(convType)) {
        kft->setFlag(false);
        RO.primaryKFs.push_back(elemaskf);
        RO.localMap.insert(seedtk, elemaskf);
      } else if (elemaskf->trackType(convType)) {
        kft->setFlag(false);
        RO.secondaryKFs.push_back(elemaskf);
        RO.localMap.insert(seedtk, elemaskf);
      }
    }  // loop on closest KFs not closer to other GSFs
  }    // loop on GSF primaries on RO
}

void PFEGammaAlgo::linkRefinableObjectPrimaryKFsToSecondaryKFs(ProtoEGObject& RO) {
  constexpr reco::PFBlockElement::TrackType convType = reco::PFBlockElement::T_FROM_GAMMACONV;
  if (_splayedblock[reco::PFBlockElement::TRACK].empty())
    return;
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend = _splayedblock[reco::PFBlockElement::TRACK].end();
  for (auto& primkf : RO.primaryKFs) {
    // don't process SC-only ROs or secondary seeded ROs
    if (primkf->trackType(convType)) {
      throw cms::Exception("PFEGammaAlgo::linkRefinableObjectPrimaryKFsToSecondaryKFs()")
          << "A KF track from conversion has been assigned as a primary!!" << std::endl;
    }
    NotCloserToOther<reco::PFBlockElement::TRACK, reco::PFBlockElement::TRACK, true> kfTrackToKFs(_currentblock,
                                                                                                  primkf);
    // get KF tracks not closer to another and not already used
    auto notlinked = std::partition(KFbegin, KFend, kfTrackToKFs);
    // attach tracks and set as used
    for (auto kft = KFbegin; kft != notlinked; ++kft) {
      const PFKFElement* elemaskf = docast(const PFKFElement*, kft->get());
      // don't care about things that aren't primaries or directly
      // associated secondary tracks
      if (elemaskf->trackType(convType)) {
        kft->setFlag(false);
        RO.secondaryKFs.push_back(elemaskf);
        RO.localMap.insert(primkf, elemaskf);
      }
    }  // loop on closest KFs not closer to other KFs
  }    // loop on KF primaries on RO
}

// try to associate the tracks to cluster elements which are not used
void PFEGammaAlgo::linkRefinableObjectPrimaryGSFTrackToECAL(ProtoEGObject& RO) {
  if (_splayedblock[reco::PFBlockElement::ECAL].empty()) {
    RO.electronClusters.push_back(nullptr);
    return;
  }
  auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();
  for (auto& primgsf : RO.primaryGSFs) {
    NotCloserToOther<reco::PFBlockElement::GSF, reco::PFBlockElement::ECAL> gsfTracksToECALs(_currentblock, primgsf);
    // get set of matching ecals not already in SC
    auto notmatched_blk = std::partition(ECALbegin, ECALend, gsfTracksToECALs);
    notmatched_blk =
        std::partition(ECALbegin, notmatched_blk, [&primgsf](auto const& x) { return compatibleEoPOut(*x, *primgsf); });
    // get set of matching ecals already in the RO
    auto notmatched_sc = std::partition(RO.ecalclusters.begin(), RO.ecalclusters.end(), gsfTracksToECALs);
    notmatched_sc = std::partition(
        RO.ecalclusters.begin(), notmatched_sc, [&primgsf](auto const& x) { return compatibleEoPOut(*x, *primgsf); });
    // look inside the SC for the ECAL cluster
    for (auto ecal = RO.ecalclusters.begin(); ecal != notmatched_sc; ++ecal) {
      const PFClusterElement* elemascluster = docast(const PFClusterElement*, ecal->get());
      FlaggedPtr<const PFClusterElement> temp(elemascluster, true);
      LOGDRESSED("PFEGammaAlgo::linkGSFTracktoECAL()") << "Found a cluster already in RO by GSF extrapolation"
                                                       << " at ECAL surface!" << std::endl
                                                       << *elemascluster << std::endl;

      RO.localMap.insert(primgsf, temp.get());
    }
    // look outside the SC for the ecal cluster
    for (auto ecal = ECALbegin; ecal != notmatched_blk; ++ecal) {
      const PFClusterElement* elemascluster = docast(const PFClusterElement*, ecal->get());
      LOGDRESSED("PFEGammaAlgo::linkGSFTracktoECAL()") << "Found a cluster not already in RO by GSF extrapolation"
                                                       << " at ECAL surface!" << std::endl
                                                       << *elemascluster << std::endl;
      if (addPFClusterToROSafe(elemascluster, RO)) {
        attachPSClusters(elemascluster, RO.ecal2ps[elemascluster]);
        RO.localMap.insert(primgsf, elemascluster);
        ecal->setFlag(false);
      }
    }
  }
}

// try to associate the tracks to cluster elements which are not used
void PFEGammaAlgo::linkRefinableObjectPrimaryGSFTrackToHCAL(ProtoEGObject& RO) {
  if (_splayedblock[reco::PFBlockElement::HCAL].empty())
    return;
  auto HCALbegin = _splayedblock[reco::PFBlockElement::HCAL].begin();
  auto HCALend = _splayedblock[reco::PFBlockElement::HCAL].end();
  for (auto& primgsf : RO.primaryGSFs) {
    NotCloserToOther<reco::PFBlockElement::GSF, reco::PFBlockElement::HCAL> gsfTracksToHCALs(_currentblock, primgsf);
    auto notmatched = std::partition(HCALbegin, HCALend, gsfTracksToHCALs);
    for (auto hcal = HCALbegin; hcal != notmatched; ++hcal) {
      const PFClusterElement* elemascluster = docast(const PFClusterElement*, hcal->get());
      FlaggedPtr<const PFClusterElement> temp(elemascluster, true);
      LOGDRESSED("PFEGammaAlgo::linkGSFTracktoECAL()")
          << "Found an HCAL cluster associated to GSF extrapolation" << std::endl;
      RO.hcalClusters.push_back(temp.get());
      RO.localMap.insert(primgsf, temp.get());
      hcal->setFlag(false);
    }
  }
}

// try to associate the tracks to cluster elements which are not used
void PFEGammaAlgo::linkRefinableObjectKFTracksToECAL(ProtoEGObject& RO) {
  if (_splayedblock[reco::PFBlockElement::ECAL].empty())
    return;
  for (auto& primkf : RO.primaryKFs)
    linkKFTrackToECAL(primkf, RO);
  for (auto& secdkf : RO.secondaryKFs)
    linkKFTrackToECAL(secdkf, RO);
}

void PFEGammaAlgo::linkKFTrackToECAL(PFKFElement const* kfflagged, ProtoEGObject& RO) {
  std::vector<FlaggedPtr<const PFClusterElement>>& currentECAL = RO.ecalclusters;
  auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();
  NotCloserToOther<reco::PFBlockElement::TRACK, reco::PFBlockElement::ECAL> kfTrackToECALs(_currentblock, kfflagged);
  NotCloserToOther<reco::PFBlockElement::GSF, reco::PFBlockElement::ECAL> kfTrackGSFToECALs(_currentblock, kfflagged);
  //get the ECAL elements not used and not closer to another KF
  auto notmatched_sc = std::partition(currentECAL.begin(), currentECAL.end(), kfTrackToECALs);
  //get subset ECAL elements not used or closer to another GSF of any type
  notmatched_sc = std::partition(currentECAL.begin(), notmatched_sc, kfTrackGSFToECALs);
  for (auto ecalitr = currentECAL.begin(); ecalitr != notmatched_sc; ++ecalitr) {
    const PFClusterElement* elemascluster = docast(const PFClusterElement*, ecalitr->get());
    FlaggedPtr<const PFClusterElement> flaggedclus(elemascluster, true);

    LOGDRESSED("PFEGammaAlgo::linkKFTracktoECAL()") << "Found a cluster already in RO by KF extrapolation"
                                                    << " at ECAL surface!" << std::endl
                                                    << *elemascluster << std::endl;
    RO.localMap.insert(elemascluster, kfflagged);
  }
  //get the ECAL elements not used and not closer to another KF
  auto notmatched_blk = std::partition(ECALbegin, ECALend, kfTrackToECALs);
  //get subset ECAL elements not used or closer to another GSF of any type
  notmatched_blk = std::partition(ECALbegin, notmatched_blk, kfTrackGSFToECALs);
  for (auto ecalitr = ECALbegin; ecalitr != notmatched_blk; ++ecalitr) {
    const PFClusterElement* elemascluster = docast(const PFClusterElement*, ecalitr->get());
    if (addPFClusterToROSafe(elemascluster, RO)) {
      attachPSClusters(elemascluster, RO.ecal2ps[elemascluster]);
      ecalitr->setFlag(false);

      LOGDRESSED("PFEGammaAlgo::linkKFTracktoECAL()") << "Found a cluster not in RO by KF extrapolation"
                                                      << " at ECAL surface!" << std::endl
                                                      << *elemascluster << std::endl;
      RO.localMap.insert(elemascluster, kfflagged);
    }
  }
}

void PFEGammaAlgo::linkRefinableObjectBremTangentsToECAL(ProtoEGObject& RO) {
  if (RO.brems.empty())
    return;
  int FirstBrem = -1;
  int TrajPos = -1;
  int lastBremTrajPos = -1;
  for (auto& brem : RO.brems) {
    bool has_clusters = false;
    TrajPos = (brem->indTrajPoint()) - 2;
    auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
    auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();
    NotCloserToOther<reco::PFBlockElement::BREM, reco::PFBlockElement::ECAL> BremToECALs(_currentblock, brem);
    // check for late brem using clusters already in the SC
    auto RSCBegin = RO.ecalclusters.begin();
    auto RSCEnd = RO.ecalclusters.end();
    auto notmatched_rsc = std::partition(RSCBegin, RSCEnd, BremToECALs);
    for (auto ecal = RSCBegin; ecal != notmatched_rsc; ++ecal) {
      float deta = std::abs((*ecal)->clusterRef()->positionREP().eta() - brem->positionAtECALEntrance().eta());
      if (deta < 0.015) {
        has_clusters = true;
        if (lastBremTrajPos == -1 || lastBremTrajPos < TrajPos) {
          lastBremTrajPos = TrajPos;
        }
        if (FirstBrem == -1 || TrajPos < FirstBrem) {  // set brem information
          FirstBrem = TrajPos;
          RO.firstBrem = TrajPos;
        }
        LOGDRESSED("PFEGammaAlgo::linkBremToECAL()") << "Found a cluster already in SC linked to brem extrapolation"
                                                     << " at ECAL surface!" << std::endl;
        RO.localMap.insert(ecal->get(), brem);
      }
    }
    // grab new clusters from the block (ensured to not be late brem)
    auto notmatched_block = std::partition(ECALbegin, ECALend, BremToECALs);
    for (auto ecal = ECALbegin; ecal != notmatched_block; ++ecal) {
      float deta = std::abs((*ecal)->clusterRef()->positionREP().eta() - brem->positionAtECALEntrance().eta());
      if (deta < 0.015) {
        has_clusters = true;
        if (lastBremTrajPos == -1 || lastBremTrajPos < TrajPos) {
          lastBremTrajPos = TrajPos;
        }
        if (FirstBrem == -1 || TrajPos < FirstBrem) {  // set brem information

          FirstBrem = TrajPos;
          RO.firstBrem = TrajPos;
        }
        const PFClusterElement* elemasclus = docast(const PFClusterElement*, ecal->get());
        if (addPFClusterToROSafe(elemasclus, RO)) {
          attachPSClusters(elemasclus, RO.ecal2ps[elemasclus]);

          RO.localMap.insert(ecal->get(), brem);
          ecal->setFlag(false);
          LOGDRESSED("PFEGammaAlgo::linkBremToECAL()") << "Found a cluster not already associated by brem extrapolation"
                                                       << " at ECAL surface!" << std::endl;
        }
      }
    }
    if (has_clusters) {
      if (RO.nBremsWithClusters == -1)
        RO.nBremsWithClusters = 0;
      ++RO.nBremsWithClusters;
    }
  }
}

void PFEGammaAlgo::linkRefinableObjectConvSecondaryKFsToSecondaryKFs(ProtoEGObject& RO) {
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend = _splayedblock[reco::PFBlockElement::TRACK].end();
  auto BeginROskfs = RO.secondaryKFs.begin();
  auto EndROskfs = RO.secondaryKFs.end();
  auto ronotconv = std::partition(BeginROskfs, EndROskfs, [](auto const& x) { return x->trackType(ConvType); });
  size_t convkfs_end = std::distance(BeginROskfs, ronotconv);
  for (size_t idx = 0; idx < convkfs_end; ++idx) {
    auto const& secKFs =
        RO.secondaryKFs;  //we want the entry at the index but we allocate to secondaryKFs in loop which invalidates all iterators, references and pointers, hence we need to get the entry fresh each time
    NotCloserToOther<reco::PFBlockElement::TRACK, reco::PFBlockElement::TRACK, true> TracksToTracks(_currentblock,
                                                                                                    secKFs[idx]);
    auto notmatched = std::partition(KFbegin, KFend, TracksToTracks);
    notmatched = std::partition(KFbegin, notmatched, [](auto const& x) { return x->trackType(ConvType); });
    for (auto kf = KFbegin; kf != notmatched; ++kf) {
      const reco::PFBlockElementTrack* elemaskf = docast(const reco::PFBlockElementTrack*, kf->get());
      RO.secondaryKFs.push_back(elemaskf);
      RO.localMap.insert(secKFs[idx], kf->get());
      kf->setFlag(false);
    }
  }
}

void PFEGammaAlgo::linkRefinableObjectECALToSingleLegConv(ProtoEGObject& RO) {
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend = _splayedblock[reco::PFBlockElement::TRACK].end();
  for (auto& ecal : RO.ecalclusters) {
    NotCloserToOther<reco::PFBlockElement::ECAL, reco::PFBlockElement::TRACK, true> ECALToTracks(_currentblock,
                                                                                                 ecal.get());
    auto notmatchedkf = std::partition(KFbegin, KFend, ECALToTracks);
    auto notconvkf = std::partition(KFbegin, notmatchedkf, [](auto const& x) { return x->trackType(ConvType); });
    // add identified KF conversion tracks
    for (auto kf = KFbegin; kf != notconvkf; ++kf) {
      const reco::PFBlockElementTrack* elemaskf = docast(const reco::PFBlockElementTrack*, kf->get());
      RO.secondaryKFs.push_back(elemaskf);
      RO.localMap.insert(ecal.get(), elemaskf);
      kf->setFlag(false);
    }
    // go through non-conv-identified kfs and check MVA to add conversions
    for (auto kf = notconvkf; kf != notmatchedkf; ++kf) {
      float mvaval = evaluateSingleLegMVA(_currentblock, primaryVertex_, (*kf)->index());
      if (mvaval > cfg_.mvaConvCut) {
        const reco::PFBlockElementTrack* elemaskf = docast(const reco::PFBlockElementTrack*, kf->get());
        RO.secondaryKFs.push_back(elemaskf);
        RO.localMap.insert(ecal.get(), elemaskf);
        kf->setFlag(false);

        RO.singleLegConversionMvaMap.emplace(elemaskf, mvaval);
      }
    }
  }
}

void PFEGammaAlgo::linkRefinableObjectSecondaryKFsToECAL(ProtoEGObject& RO) {
  auto ECALbegin = _splayedblock[reco::PFBlockElement::ECAL].begin();
  auto ECALend = _splayedblock[reco::PFBlockElement::ECAL].end();
  for (auto& skf : RO.secondaryKFs) {
    NotCloserToOther<reco::PFBlockElement::TRACK, reco::PFBlockElement::ECAL, false> TracksToECALwithCut(
        _currentblock, skf, 1.5f);
    auto notmatched = std::partition(ECALbegin, ECALend, TracksToECALwithCut);
    for (auto ecal = ECALbegin; ecal != notmatched; ++ecal) {
      const reco::PFBlockElementCluster* elemascluster = docast(const reco::PFBlockElementCluster*, ecal->get());
      if (addPFClusterToROSafe(elemascluster, RO)) {
        attachPSClusters(elemascluster, RO.ecal2ps[elemascluster]);
        RO.localMap.insert(skf, elemascluster);
        ecal->setFlag(false);
      }
    }
  }
}

PFEGammaAlgo::EgammaObjects PFEGammaAlgo::fillPFCandidates(const std::list<PFEGammaAlgo::ProtoEGObject>& ROs) {
  EgammaObjects output;

  // reserve output collections
  output.candidates.reserve(ROs.size());
  output.candidateExtras.reserve(ROs.size());
  output.refinedSuperClusters.reserve(ROs.size());

  for (auto& RO : ROs) {
    if (RO.ecalclusters.empty() && !cfg_.produceEGCandsWithNoSuperCluster)
      continue;

    reco::PFCandidate cand;
    reco::PFCandidateEGammaExtra xtra;
    if (!RO.primaryGSFs.empty() || !RO.primaryKFs.empty()) {
      cand.setPdgId(-11);  // anything with a primary track is an electron
    } else {
      cand.setPdgId(22);  // anything with no primary track is a photon
    }
    if (!RO.primaryKFs.empty()) {
      cand.setCharge(RO.primaryKFs[0]->trackRef()->charge());
      xtra.setKfTrackRef(RO.primaryKFs[0]->trackRef());
      cand.setTrackRef(RO.primaryKFs[0]->trackRef());
      cand.addElementInBlock(_currentblock, RO.primaryKFs[0]->index());
    }
    if (!RO.primaryGSFs.empty()) {
      cand.setCharge(RO.primaryGSFs[0]->GsftrackRef()->chargeMode());
      xtra.setGsfTrackRef(RO.primaryGSFs[0]->GsftrackRef());
      cand.setGsfTrackRef(RO.primaryGSFs[0]->GsftrackRef());
      cand.addElementInBlock(_currentblock, RO.primaryGSFs[0]->index());
    }
    if (RO.parentSC) {
      xtra.setSuperClusterPFECALRef(RO.parentSC->superClusterRef());
      // we'll set to the refined supercluster back up in the producer
      cand.setSuperClusterRef(RO.parentSC->superClusterRef());
      xtra.setSuperClusterRef(RO.parentSC->superClusterRef());
      cand.addElementInBlock(_currentblock, RO.parentSC->index());
    }
    // add brems
    for (const auto& brem : RO.brems) {
      cand.addElementInBlock(_currentblock, brem->index());
    }
    // add clusters and ps clusters
    for (const auto& ecal : RO.ecalclusters) {
      const PFClusterElement* clus = ecal.get();
      cand.addElementInBlock(_currentblock, clus->index());
      if (RO.ecal2ps.count(clus)) {
        for (auto& psclus : RO.ecal2ps.at(clus)) {
          cand.addElementInBlock(_currentblock, psclus->index());
        }
      }
    }
    // add secondary tracks
    for (const auto& kf : RO.secondaryKFs) {
      cand.addElementInBlock(_currentblock, kf->index());
      const reco::ConversionRefVector& convrefs = kf->convRefs();
      bool no_conv_ref = true;
      for (const auto& convref : convrefs) {
        if (convref.isNonnull() && convref.isAvailable()) {
          xtra.addConversionRef(convref);
          no_conv_ref = false;
        }
      }
      if (no_conv_ref) {
        //single leg conversions

        //look for stored mva value in map or else recompute
        const auto& mvavalmapped = RO.singleLegConversionMvaMap.find(kf);
        //FIXME: Abuse single mva value to store both provenance and single leg mva score
        //by storing 3.0 + mvaval
        float mvaval = (mvavalmapped != RO.singleLegConversionMvaMap.end()
                            ? mvavalmapped->second
                            : 3.0 + evaluateSingleLegMVA(_currentblock, primaryVertex_, kf->index()));

        xtra.addSingleLegConvTrackRefMva(std::make_pair(kf->trackRef(), mvaval));
      }
    }

    // build the refined supercluster from those clusters left in the cand
    output.refinedSuperClusters.push_back(buildRefinedSuperCluster(RO));

    //*TODO* cluster time is not reliable at the moment, so only use track timing
    float trkTime = 0, trkTimeErr = -1;
    if (!RO.primaryGSFs.empty() && RO.primaryGSFs[0]->isTimeValid()) {
      trkTime = RO.primaryGSFs[0]->time();
      trkTimeErr = RO.primaryGSFs[0]->timeError();
    } else if (!RO.primaryKFs.empty() && RO.primaryKFs[0]->isTimeValid()) {
      trkTime = RO.primaryKFs[0]->time();
      trkTimeErr = RO.primaryKFs[0]->timeError();
    }
    if (trkTimeErr >= 0) {
      cand.setTime(trkTime, trkTimeErr);
    }

    const reco::SuperCluster& the_sc = output.refinedSuperClusters.back();
    // with the refined SC in hand we build a naive candidate p4
    // and set the candidate ECAL position to either the barycenter of the
    // supercluster (if super-cluster present) or the seed of the
    // new SC generated by the EGAlgo
    const double scE = the_sc.energy();
    if (scE != 0.0) {
      const math::XYZPoint& seedPos = the_sc.seed()->position();
      math::XYZVector egDir = the_sc.position() - primaryVertex_.position();
      egDir = egDir.Unit();
      cand.setP4(math::XYZTLorentzVector(scE * egDir.x(), scE * egDir.y(), scE * egDir.z(), scE));
      math::XYZPointF ecalPOS_f(seedPos.x(), seedPos.y(), seedPos.z());
      cand.setPositionAtECALEntrance(ecalPOS_f);
      cand.setEcalEnergy(the_sc.rawEnergy(), the_sc.energy());
    } else if (cfg_.produceEGCandsWithNoSuperCluster && !RO.primaryGSFs.empty()) {
      const PFGSFElement* gsf = RO.primaryGSFs[0];
      const reco::GsfTrackRef& gref = gsf->GsftrackRef();
      math::XYZTLorentzVector p4(gref->pxMode(), gref->pyMode(), gref->pzMode(), gref->pMode());
      cand.setP4(p4);
      cand.setPositionAtECALEntrance(gsf->positionAtECALEntrance());
    } else if (cfg_.produceEGCandsWithNoSuperCluster && !RO.primaryKFs.empty()) {
      const PFKFElement* kf = RO.primaryKFs[0];
      reco::TrackRef kref = RO.primaryKFs[0]->trackRef();
      math::XYZTLorentzVector p4(kref->px(), kref->py(), kref->pz(), kref->p());
      cand.setP4(p4);
      cand.setPositionAtECALEntrance(kf->positionAtECALEntrance());
    }
    const float eleMVAValue = calculateEleMVA(RO, xtra);
    fillExtraInfo(RO, xtra);
    //std::cout << "PFEG eleMVA: " << eleMVAValue << std::endl;
    xtra.setMVA(eleMVAValue);
    cand.set_mva_e_pi(eleMVAValue);
    output.candidates.push_back(cand);
    output.candidateExtras.push_back(xtra);
  }

  return output;
}

float PFEGammaAlgo::calculateEleMVA(const PFEGammaAlgo::ProtoEGObject& ro, reco::PFCandidateEGammaExtra& xtra) const {
  if (ro.primaryGSFs.empty()) {
    return -2.0f;
  }
  const PFGSFElement* gsfElement = ro.primaryGSFs.front();
  const PFKFElement* kfElement = nullptr;
  if (!ro.primaryKFs.empty()) {
    kfElement = ro.primaryKFs.front();
  }
  auto const& refGsf = gsfElement->GsftrackRef();
  reco::TrackRef refKf;
  constexpr float mEl = 0.000511;
  const double eInGsf = std::hypot(refGsf->pMode(), mEl);
  double dEtGsfEcal = 1e6;
  double sigmaEtaEta = 1e-14;
  const double eneHcalGsf =
      std::accumulate(ro.hcalClusters.begin(), ro.hcalClusters.end(), 0.0, [](const double a, auto const& b) {
        return a + b->clusterRef()->energy();
      });
  if (!ro.primaryKFs.empty()) {
    refKf = ro.primaryKFs.front()->trackRef();
  }
  const double eOutGsf = gsfElement->Pout().t();
  const double etaOutGsf = gsfElement->positionAtECALEntrance().eta();
  double firstEcalGsfEnergy{0.0};
  double otherEcalGsfEnergy{0.0};
  double ecalBremEnergy{0.0};
  //shower shape of cluster closest to gsf track
  std::vector<const reco::PFCluster*> gsfCluster;
  for (const auto& ecal : ro.ecalclusters) {
    const double cenergy = ecal->clusterRef()->correctedEnergy();
    bool hasgsf = ro.localMap.contains(gsfElement, ecal.get());
    bool haskf = ro.localMap.contains(kfElement, ecal.get());
    bool hasbrem = false;
    for (const auto& brem : ro.brems) {
      if (ro.localMap.contains(brem, ecal.get())) {
        hasbrem = true;
      }
    }
    if (hasbrem && ecal.get() != ro.electronClusters[0]) {
      ecalBremEnergy += cenergy;
    }
    if (!hasbrem && ecal.get() != ro.electronClusters[0]) {
      if (hasgsf)
        otherEcalGsfEnergy += cenergy;
      if (haskf)
        ecalBremEnergy += cenergy;  // from conv. brem!
      if (!(hasgsf || haskf))
        otherEcalGsfEnergy += cenergy;  // stuff from SC
    }
  }

  if (ro.electronClusters[0]) {
    reco::PFClusterRef cref = ro.electronClusters[0]->clusterRef();
    xtra.setGsfElectronClusterRef(_currentblock, *(ro.electronClusters[0]));
    firstEcalGsfEnergy = cref->correctedEnergy();
    dEtGsfEcal = cref->positionREP().eta() - etaOutGsf;
    gsfCluster.push_back(cref.get());
    PFClusterWidthAlgo pfwidth(gsfCluster);
    sigmaEtaEta = pfwidth.pflowSigmaEtaEta();
  }

  // brem sequence information
  float firstBrem{-1.0f};
  float earlyBrem{-1.0f};
  float lateBrem{-1.0f};
  if (ro.nBremsWithClusters > 0) {
    firstBrem = ro.firstBrem;
    earlyBrem = ro.firstBrem < 4 ? 1.0f : 0.0f;
    lateBrem = ro.lateBrem == 1 ? 1.0f : 0.0f;
  }
  xtra.setEarlyBrem(earlyBrem);
  xtra.setLateBrem(lateBrem);
  if (firstEcalGsfEnergy > 0.0) {
    if (refGsf.isNonnull()) {
      xtra.setGsfTrackPout(gsfElement->Pout());
      // normalization observables
      const float ptGsf = refGsf->ptMode();
      const float etaGsf = refGsf->etaMode();
      // tracking observables
      const double ptModeErrorGsf = refGsf->ptModeError();
      float ptModeErrOverPtGsf = (ptModeErrorGsf > 0. ? ptModeErrorGsf / ptGsf : 1.0);
      float chi2Gsf = refGsf->normalizedChi2();
      float dPtOverPtGsf = (ptGsf - gsfElement->Pout().pt()) / ptGsf;
      // kalman filter vars
      float nHitKf = refKf.isNonnull() ? refKf->hitPattern().trackerLayersWithMeasurement() : 0;
      float chi2Kf = refKf.isNonnull() ? refKf->normalizedChi2() : -0.01;

      //tracker + calorimetry observables
      float eTotPinMode = (firstEcalGsfEnergy + otherEcalGsfEnergy + ecalBremEnergy) / eInGsf;
      float eGsfPoutMode = firstEcalGsfEnergy / eOutGsf;
      float eTotBremPinPoutMode = (ecalBremEnergy + otherEcalGsfEnergy) / (eInGsf - eOutGsf);
      float dEtaGsfEcalClust = std::abs(dEtGsfEcal);
      float logSigmaEtaEta = std::log(sigmaEtaEta);
      float hOverHe = eneHcalGsf / (eneHcalGsf + firstEcalGsfEnergy);

      xtra.setDeltaEta(dEtaGsfEcalClust);
      xtra.setSigmaEtaEta(sigmaEtaEta);
      xtra.setHadEnergy(eneHcalGsf);

      // Apply bounds to variables and calculate MVA
      dPtOverPtGsf = std::clamp(dPtOverPtGsf, -0.2f, 1.0f);
      ptModeErrOverPtGsf = std::min(ptModeErrOverPtGsf, 0.3f);
      chi2Gsf = std::min(chi2Gsf, 10.0f);
      chi2Kf = std::min(chi2Kf, 10.0f);
      eTotPinMode = std::clamp(eTotPinMode, 0.0f, 5.0f);
      eGsfPoutMode = std::clamp(eGsfPoutMode, 0.0f, 5.0f);
      eTotBremPinPoutMode = std::clamp(eTotBremPinPoutMode, 0.0f, 5.0f);
      dEtaGsfEcalClust = std::min(dEtaGsfEcalClust, 0.1f);
      logSigmaEtaEta = std::max(logSigmaEtaEta, -14.0f);

      // not used for moment, weird behavior of variable
      //float dPtOverPtKf = refKf.isNonnull() ? (refKf->pt() - refKf->outerPt())/refKf->pt() : -0.01;
      //dPtOverPtKf       = std::clamp(dPtOverPtKf,-0.2f, 1.0f);

      /*
 *      To be used for debugging:
 *      pretty-print the PFEgamma electron MVA input variables
 *
 *      std::cout << " **** PFEG BDT observables ****" << endl;
 *      std::cout << " < Normalization > " << endl;
 *      std::cout << " ptGsf " << ptGsf << " Pin " << eInGsf
 *        << " Pout " << eOutGsf << " etaGsf " << etaGsf << endl;
 *      std::cout << " < PureTracking > " << endl;
 *      std::cout << " ptModeErrOverPtGsf " << ptModeErrOverPtGsf 
 *        << " dPtOverPtGsf " << dPtOverPtGsf
 *        << " chi2Gsf " << chi2Gsf
 *        << " nhit_gsf " << nhit_gsf
 *        << " dPtOverPtKf " << dPtOverPtKf
 *        << " chi2Kf " << chi2Kf 
 *        << " nHitKf " << nHitKf <<  endl;
 *      std::cout << " < track-ecal-hcal-ps " << endl;
 *      std::cout << " eTotPinMode " << eTotPinMode 
 *        << " eGsfPoutMode " << eGsfPoutMode
 *        << " eTotBremPinPoutMode " << eTotBremPinPoutMode
 *        << " dEtaGsfEcalClust " << dEtaGsfEcalClust 
 *        << " logSigmaEtaEta " << logSigmaEtaEta
 *        << " hOverHe " << hOverHe << " Hcal energy " << eneHcalGsf
 *        << " lateBrem " << lateBrem
 *        << " firstBrem " << firstBrem << endl;
 */

      float vars[] = {std::log(ptGsf),
                      etaGsf,
                      ptModeErrOverPtGsf,
                      dPtOverPtGsf,
                      chi2Gsf,
                      nHitKf,
                      chi2Kf,
                      eTotPinMode,
                      eGsfPoutMode,
                      eTotBremPinPoutMode,
                      dEtaGsfEcalClust,
                      logSigmaEtaEta,
                      hOverHe,
                      lateBrem,
                      firstBrem};

      return gbrForests_.ele_->GetAdaBoostClassifier(vars);
    }
  }
  return -2.0f;
}

void PFEGammaAlgo::fillExtraInfo(const ProtoEGObject& RO, reco::PFCandidateEGammaExtra& xtra) {
  // add tracks associated to clusters that are not T_FROM_GAMMACONV
  // info about single-leg convs is already save, so just veto in loops
  auto KFbegin = _splayedblock[reco::PFBlockElement::TRACK].begin();
  auto KFend = _splayedblock[reco::PFBlockElement::TRACK].end();
  for (auto& ecal : RO.ecalclusters) {
    NotCloserToOther<reco::PFBlockElement::ECAL, reco::PFBlockElement::TRACK, true> ECALToTracks(_currentblock,
                                                                                                 ecal.get());
    auto notmatchedkf = std::partition(KFbegin, KFend, ECALToTracks);
    auto notconvkf = std::partition(KFbegin, notmatchedkf, [](auto const& x) { return x->trackType(ConvType); });
    // go through non-conv-identified kfs and check MVA to add conversions
    for (auto kf = notconvkf; kf != notmatchedkf; ++kf) {
      const reco::PFBlockElementTrack* elemaskf = docast(const reco::PFBlockElementTrack*, kf->get());
      xtra.addExtraNonConvTrack(_currentblock, *elemaskf);
    }
  }
}

// currently stolen from PFECALSuperClusterAlgo, we should
// try to factor this correctly since the operation is the same in
// both places...
reco::SuperCluster PFEGammaAlgo::buildRefinedSuperCluster(const PFEGammaAlgo::ProtoEGObject& RO) {
  if (RO.ecalclusters.empty()) {
    return reco::SuperCluster(0.0, math::XYZPoint(0, 0, 0));
  }

  bool isEE = false;
  // need the vector of raw pointers for a PF width class
  std::vector<const reco::PFCluster*> bare_ptrs;
  // calculate necessary parameters and build the SC
  double posX(0), posY(0), posZ(0), rawSCEnergy(0), corrSCEnergy(0), corrPSEnergy(0), ps1_energy(0.0), ps2_energy(0.0);
  for (auto& clus : RO.ecalclusters) {
    double ePS1 = 0;
    double ePS2 = 0;
    isEE = PFLayer::ECAL_ENDCAP == clus->clusterRef()->layer();
    auto clusptr = edm::refToPtr<reco::PFClusterCollection>(clus->clusterRef());
    bare_ptrs.push_back(clusptr.get());

    const double cluseraw = clusptr->energy();
    double cluscalibe = clusptr->correctedEnergy();
    const math::XYZPoint& cluspos = clusptr->position();
    posX += cluseraw * cluspos.X();
    posY += cluseraw * cluspos.Y();
    posZ += cluseraw * cluspos.Z();
    // update EE calibrated super cluster energies
    if (isEE && RO.ecal2ps.count(clus.get())) {
      const auto& psclusters = RO.ecal2ps.at(clus.get());

      std::vector<reco::PFCluster const*> psClusterPointers;
      for (auto const& psc : psclusters) {
        psClusterPointers.push_back(psc->clusterRef().get());
      }
      auto calibratedEnergies = thePFEnergyCalibration_.calibrateEndcapClusterEnergies(
          *clusptr, psClusterPointers, channelStatus_, cfg_.applyCrackCorrections);
      cluscalibe = calibratedEnergies.clusterEnergy;
      ePS1 = calibratedEnergies.ps1Energy;
      ePS2 = calibratedEnergies.ps2Energy;
    }
    if (ePS1 == -1.)
      ePS1 = 0;
    if (ePS2 == -1.)
      ePS2 = 0;

    rawSCEnergy += cluseraw;
    corrSCEnergy += cluscalibe;
    ps1_energy += ePS1;
    ps2_energy += ePS2;
    corrPSEnergy += ePS1 + ePS2;
  }
  posX /= rawSCEnergy;
  posY /= rawSCEnergy;
  posZ /= rawSCEnergy;

  // now build the supercluster
  reco::SuperCluster new_sc(corrSCEnergy, math::XYZPoint(posX, posY, posZ));

  auto clusptr = edm::refToPtr<reco::PFClusterCollection>(RO.ecalclusters.front()->clusterRef());
  new_sc.setCorrectedEnergy(corrSCEnergy);
  new_sc.setSeed(clusptr);
  new_sc.setPreshowerEnergyPlane1(ps1_energy);
  new_sc.setPreshowerEnergyPlane2(ps2_energy);
  new_sc.setPreshowerEnergy(corrPSEnergy);
  for (const auto& clus : RO.ecalclusters) {
    clusptr = edm::refToPtr<reco::PFClusterCollection>(clus->clusterRef());
    new_sc.addCluster(clusptr);
    auto& hits_and_fractions = clusptr->hitsAndFractions();
    for (auto& hit_and_fraction : hits_and_fractions) {
      new_sc.addHitAndFraction(hit_and_fraction.first, hit_and_fraction.second);
    }
    // put the preshower stuff back in later
    if (RO.ecal2ps.count(clus.get())) {
      const auto& cluspsassociation = RO.ecal2ps.at(clus.get());
      // EE rechits should be uniquely matched to sets of pre-shower
      // clusters at this point, so we throw an exception if otherwise
      // now wrapped in EDM debug flags
      for (const auto& pscluselem : cluspsassociation) {
        edm::Ptr<reco::PFCluster> psclus = edm::refToPtr<reco::PFClusterCollection>(pscluselem->clusterRef());
#ifdef PFFLOW_DEBUG
        auto found_pscluster =
            std::find(new_sc.preshowerClustersBegin(), new_sc.preshowerClustersEnd(), reco::CaloClusterPtr(psclus));
        if (found_pscluster == new_sc.preshowerClustersEnd()) {
#endif
          new_sc.addPreshowerCluster(psclus);
#ifdef PFFLOW_DEBUG
        } else {
          throw cms::Exception("PFECALSuperClusterAlgo::buildSuperCluster")
              << "Found a PS cluster matched to more than one EE cluster!" << std::endl
              << std::hex << psclus.get() << " == " << found_pscluster->get() << std::dec << std::endl;
        }
#endif
      }
    }
  }

  // calculate linearly weighted cluster widths
  PFClusterWidthAlgo pfwidth(bare_ptrs);
  new_sc.setEtaWidth(pfwidth.pflowEtaWidth());
  new_sc.setPhiWidth(pfwidth.pflowPhiWidth());

  // cache the value of the raw energy
  new_sc.rawEnergy();

  return new_sc;
}

void PFEGammaAlgo::unlinkRefinableObjectKFandECALWithBadEoverP(ProtoEGObject& RO) {
  // this only means something for ROs with a primary GSF track
  if (RO.primaryGSFs.empty())
    return;
  // need energy sums to tell if we've added crap or not
  const double Pin_gsf = RO.primaryGSFs.front()->GsftrackRef()->pMode();
  const double gsfOuterEta = RO.primaryGSFs.front()->positionAtECALEntrance().Eta();
  double tot_ecal = 0.0;
  std::vector<double> min_brem_dists;
  std::vector<double> closest_brem_eta;
  // first get the total ecal energy (we should replace this with a cache)
  for (const auto& ecal : RO.ecalclusters) {
    tot_ecal += ecal->clusterRef()->correctedEnergy();
    // we also need to look at the minimum distance to brems
    // since energetic brems will be closer to the brem than the track
    double min_brem_dist = 5000.0;
    double eta = -999.0;
    for (const auto& brem : RO.brems) {
      const float dist = _currentblock->dist(brem->index(), ecal->index(), _currentlinks, reco::PFBlock::LINKTEST_ALL);
      if (dist < min_brem_dist && dist != -1.0f) {
        min_brem_dist = dist;
        eta = brem->positionAtECALEntrance().Eta();
      }
    }
    min_brem_dists.push_back(min_brem_dist);
    closest_brem_eta.push_back(eta);
  }

  // loop through the ECAL clusters and remove ECAL clusters matched to
  // secondary track either in *or* out of the SC if the E/pin is bad
  for (auto secd_kf = RO.secondaryKFs.begin(); secd_kf != RO.secondaryKFs.end(); ++secd_kf) {
    reco::TrackRef trkRef = (*secd_kf)->trackRef();
    const float secpin = (*secd_kf)->trackRef()->p();
    bool remove_this_kf = false;
    for (auto ecal = RO.ecalclusters.begin(); ecal != RO.ecalclusters.end(); ++ecal) {
      size_t bremidx = std::distance(RO.ecalclusters.begin(), ecal);
      const float minbremdist = min_brem_dists[bremidx];
      const double ecalenergy = (*ecal)->clusterRef()->correctedEnergy();
      const double Epin = ecalenergy / secpin;
      const double detaGsf = std::abs(gsfOuterEta - (*ecal)->clusterRef()->positionREP().Eta());
      const double detaBrem = std::abs(closest_brem_eta[bremidx] - (*ecal)->clusterRef()->positionREP().Eta());

      bool kf_matched = RO.localMap.contains(ecal->get(), *secd_kf);

      const float tkdist =
          _currentblock->dist((*secd_kf)->index(), (*ecal)->index(), _currentlinks, reco::PFBlock::LINKTEST_ALL);

      // do not reject this track if it is closer to a brem than the
      // secondary track, or if it lies in the delta-eta plane with the
      // gsf track or if it is in the dEta plane with the brems
      if (Epin > 3 && kf_matched && tkdist != -1.0f && tkdist < minbremdist && detaGsf > 0.05 && detaBrem > 0.015) {
        double res_with = std::abs((tot_ecal - Pin_gsf) / Pin_gsf);
        double res_without = std::abs((tot_ecal - ecalenergy - Pin_gsf) / Pin_gsf);
        if (res_without < res_with) {
          LOGDRESSED("PFEGammaAlgo") << " REJECTED_RES totenergy " << tot_ecal << " Pin_gsf " << Pin_gsf
                                     << " cluster to secondary " << ecalenergy << " res_with " << res_with
                                     << " res_without " << res_without << std::endl;
          tot_ecal -= ecalenergy;
          remove_this_kf = true;
          ecal = RO.ecalclusters.erase(ecal);
          if (ecal == RO.ecalclusters.end())
            break;
        }
      }
    }
    if (remove_this_kf) {
      secd_kf = RO.secondaryKFs.erase(secd_kf);
      if (secd_kf == RO.secondaryKFs.end())
        break;
    }
  }
}

void PFEGammaAlgo::unlinkRefinableObjectKFandECALMatchedToHCAL(ProtoEGObject& RO,
                                                               bool removeFreeECAL,
                                                               bool removeSCEcal) {
  std::vector<bool> cluster_in_sc;
  auto ecal_begin = RO.ecalclusters.begin();
  auto ecal_end = RO.ecalclusters.end();
  auto hcal_begin = _splayedblock[reco::PFBlockElement::HCAL].begin();
  auto hcal_end = _splayedblock[reco::PFBlockElement::HCAL].end();
  for (auto secd_kf = RO.secondaryKFs.begin(); secd_kf != RO.secondaryKFs.end(); ++secd_kf) {
    bool remove_this_kf = false;
    NotCloserToOther<reco::PFBlockElement::TRACK, reco::PFBlockElement::HCAL> tracksToHCALs(_currentblock, *secd_kf);
    reco::TrackRef trkRef = (*secd_kf)->trackRef();

    bool goodTrack = PFTrackAlgoTools::isGoodForEGM(trkRef->algo());
    const float secpin = trkRef->p();

    for (auto ecal = ecal_begin; ecal != ecal_end; ++ecal) {
      const double ecalenergy = (*ecal)->clusterRef()->correctedEnergy();
      // first check if the cluster is in the SC (use dist calc for fastness)
      const size_t clus_idx = std::distance(ecal_begin, ecal);
      if (cluster_in_sc.size() < clus_idx + 1) {
        float dist = -1.0f;
        if (RO.parentSC) {
          dist = _currentblock->dist((*secd_kf)->index(), (*ecal)->index(), _currentlinks, reco::PFBlock::LINKTEST_ALL);
        }
        cluster_in_sc.push_back(dist != -1.0f);
      }

      // if we've found a secondary KF that matches this ecal cluster
      // now we see if it is matched to HCAL
      // if it is matched to an HCAL cluster we take different
      // actions if the cluster was in an SC or not
      if (RO.localMap.contains(ecal->get(), *secd_kf)) {
        auto hcal_matched = std::partition(hcal_begin, hcal_end, tracksToHCALs);
        for (auto hcalclus = hcal_begin; hcalclus != hcal_matched; ++hcalclus) {
          const reco::PFBlockElementCluster* clusthcal =
              dynamic_cast<const reco::PFBlockElementCluster*>(hcalclus->get());
          const double hcalenergy = clusthcal->clusterRef()->energy();
          const double hpluse = ecalenergy + hcalenergy;
          const bool isHoHE = ((hcalenergy / hpluse) > 0.1 && goodTrack);
          const bool isHoE = (hcalenergy > ecalenergy);
          const bool isPoHE = (secpin > hpluse);
          if (cluster_in_sc[clus_idx]) {
            if (isHoE || isPoHE) {
              LOGDRESSED("PFEGammaAlgo") << "REJECTED TRACK FOR H/E or P/(H+E), CLUSTER IN SC"
                                         << " H/H+E " << (hcalenergy / hpluse) << " H/E " << (hcalenergy > ecalenergy)
                                         << " P/(H+E) " << (secpin / hpluse) << " HCAL ENE " << hcalenergy
                                         << " ECAL ENE " << ecalenergy << " secPIN " << secpin << " Algo Track "
                                         << trkRef->algo() << std::endl;
              remove_this_kf = true;
            }
          } else {
            if (isHoHE) {
              LOGDRESSED("PFEGammaAlgo") << "REJECTED TRACK FOR H/H+E, CLUSTER NOT IN SC"
                                         << " H/H+E " << (hcalenergy / hpluse) << " H/E " << (hcalenergy > ecalenergy)
                                         << " P/(H+E) " << (secpin / hpluse) << " HCAL ENE " << hcalenergy
                                         << " ECAL ENE " << ecalenergy << " secPIN " << secpin << " Algo Track "
                                         << trkRef->algo() << std::endl;
              remove_this_kf = true;
            }
          }
        }
      }
    }
    if (remove_this_kf) {
      secd_kf = RO.secondaryKFs.erase(secd_kf);
      if (secd_kf == RO.secondaryKFs.end())
        break;
    }
  }
}

bool PFEGammaAlgo::isPrimaryTrack(const reco::PFBlockElementTrack& KfEl, const reco::PFBlockElementGsfTrack& GsfEl) {
  bool isPrimary = false;

  const GsfPFRecTrackRef& gsfPfRef = GsfEl.GsftrackRefPF();

  if (gsfPfRef.isNonnull()) {
    const PFRecTrackRef& kfPfRef = KfEl.trackRefPF();
    PFRecTrackRef kfPfRef_fromGsf = (*gsfPfRef).kfPFRecTrackRef();
    if (kfPfRef.isNonnull() && kfPfRef_fromGsf.isNonnull()) {
      reco::TrackRef kfref = (*kfPfRef).trackRef();
      reco::TrackRef kfref_fromGsf = (*kfPfRef_fromGsf).trackRef();
      if (kfref.isNonnull() && kfref_fromGsf.isNonnull()) {
        if (kfref == kfref_fromGsf)
          isPrimary = true;
      }
    }
  }

  return isPrimary;
}
