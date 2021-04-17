#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class TrackAndHCALLinker : public BlockElementLinkerBase {
public:
  TrackAndHCALLinker(const edm::ParameterSet& conf)
      : BlockElementLinkerBase(conf),
        useKDTree_(conf.getParameter<bool>("useKDTree")),
        trajectoryLayerEntranceString_(conf.getParameter<std::string>("trajectoryLayerEntrance")),
        trajectoryLayerExitString_(conf.getParameter<std::string>("trajectoryLayerExit")),
        debug_(conf.getUntrackedParameter<bool>("debug", false)) {
    // convert TrajectoryLayers info from string to enum
    trajectoryLayerEntrance_ = reco::PFTrajectoryPoint::layerTypeByName(trajectoryLayerEntranceString_);
    trajectoryLayerExit_ = reco::PFTrajectoryPoint::layerTypeByName(trajectoryLayerExitString_);
    // make sure the requested setting is supported
    assert((trajectoryLayerEntrance_ == reco::PFTrajectoryPoint::HCALEntrance &&
            trajectoryLayerExit_ == reco::PFTrajectoryPoint::HCALExit) ||
           (trajectoryLayerEntrance_ == reco::PFTrajectoryPoint::HCALEntrance &&
            trajectoryLayerExit_ == reco::PFTrajectoryPoint::Unknown) ||
           (trajectoryLayerEntrance_ == reco::PFTrajectoryPoint::VFcalEntrance &&
            trajectoryLayerExit_ == reco::PFTrajectoryPoint::Unknown));
    // flag if exit layer should be checked or not
    checkExit_ = trajectoryLayerExit_ != reco::PFTrajectoryPoint::Unknown;
  }

  bool linkPrefilter(const reco::PFBlockElement*, const reco::PFBlockElement*) const override;

  double testLink(const reco::PFBlockElement*, const reco::PFBlockElement*) const override;

private:
  bool useKDTree_;
  std::string trajectoryLayerEntranceString_;
  std::string trajectoryLayerExitString_;
  reco::PFTrajectoryPoint::LayerType trajectoryLayerEntrance_;
  reco::PFTrajectoryPoint::LayerType trajectoryLayerExit_;
  bool debug_;
  bool checkExit_;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, TrackAndHCALLinker, "TrackAndHCALLinker");

bool TrackAndHCALLinker::linkPrefilter(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const {
  bool result = false;
  // Track-HCAL KDTree multilinks are stored to track's elem
  switch (elem1->type()) {
    case reco::PFBlockElement::TRACK:
      result = (elem1->isMultilinksValide(elem2->type()) && !elem1->getMultilinks(elem2->type()).empty() &&
                elem2->isMultilinksValide(elem1->type()));
      break;
    case reco::PFBlockElement::HCAL:
      result = (elem2->isMultilinksValide(elem1->type()) && !elem2->getMultilinks(elem1->type()).empty() &&
                elem1->isMultilinksValide(elem2->type()));
    default:
      break;
  }
  return (useKDTree_ ? result : true);
}

double TrackAndHCALLinker::testLink(const reco::PFBlockElement* elem1, const reco::PFBlockElement* elem2) const {
  const reco::PFBlockElementCluster* hcalelem(nullptr);
  const reco::PFBlockElementTrack* tkelem(nullptr);
  double dist(-1.0);
  if (elem1->type() < elem2->type()) {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem1);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem2);
  } else {
    tkelem = static_cast<const reco::PFBlockElementTrack*>(elem2);
    hcalelem = static_cast<const reco::PFBlockElementCluster*>(elem1);
  }
  const reco::PFRecTrackRef& trackref = tkelem->trackRefPF();
  const reco::PFClusterRef& clusterref = hcalelem->clusterRef();
  const reco::PFCluster::REPPoint& hcalreppos = clusterref->positionREP();
  const reco::PFTrajectoryPoint& tkAtHCALEnt = trackref->extrapolatedPoint(trajectoryLayerEntrance_);
  const reco::PFCluster::REPPoint& tkreppos = tkAtHCALEnt.positionREP();
  // Check exit point
  double dHEta = 0.;
  double dHPhi = 0.;
  double dRHCALEx = 0.;
  if (checkExit_) {
    const reco::PFTrajectoryPoint& tkAtHCALEx = trackref->extrapolatedPoint(trajectoryLayerExit_);
    dHEta = (tkAtHCALEx.positionREP().Eta() - tkAtHCALEnt.positionREP().Eta());
    dHPhi = reco::deltaPhi(tkAtHCALEx.positionREP().Phi(), tkAtHCALEnt.positionREP().Phi());
    dRHCALEx = tkAtHCALEx.position().R();
  }

  // Check if the linking has been done using the KDTree algo
  // Glowinski & Gouzevitch
  if (useKDTree_ && tkelem->isMultilinksValide(hcalelem->type())) {  //KDTree Algo
    const reco::PFMultilinksType& multilinks = tkelem->getMultilinks(hcalelem->type());
    const double hcalphi = hcalreppos.Phi();
    const double hcaleta = hcalreppos.Eta();

    // Check if the link Track/Hcal exist
    reco::PFMultilinksType::const_iterator mlit = multilinks.begin();
    for (; mlit != multilinks.end(); ++mlit)
      if ((mlit->first == hcalphi) && (mlit->second == hcaleta))
        break;

    // If the link exist, we fill dist and linktest.

    if (mlit != multilinks.end()) {
      // when checkExit_ is false
      if (!checkExit_) {
        dist = LinkByRecHit::computeDist(hcaleta, hcalphi, tkreppos.Eta(), tkreppos.Phi());
      }
      // when checkExit_ is true
      else {
        //special case ! A looper  can exit the barrel inwards and hit the endcap
        //In this case calculate the distance based on the first crossing since
        //the looper will probably never make it to the endcap
        if (dRHCALEx < tkAtHCALEnt.position().R()) {
          dist = LinkByRecHit::computeDist(hcaleta, hcalphi, tkreppos.Eta(), tkreppos.Phi());
          edm::LogWarning("TrackHCALLinker ")
              << "Special case of linking with track hitting HCAL and looping back in the tracker ";
        } else {
          dist =
              LinkByRecHit::computeDist(hcaleta, hcalphi, tkreppos.Eta() + 0.1 * dHEta, tkreppos.Phi() + 0.1 * dHPhi);
        }
      }  // checkExit_
    }    // multilinks

  } else {  // Old algorithm
    if (tkAtHCALEnt.isValid())
      dist = LinkByRecHit::testTrackAndClusterByRecHit(*trackref, *clusterref, false, debug_);
  }
  return dist;
}
