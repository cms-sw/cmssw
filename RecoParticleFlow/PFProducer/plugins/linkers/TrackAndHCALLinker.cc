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
        _useKDTree(conf.getParameter<bool>("useKDTree")),
        _trajectoryLayerEntranceString(conf.getParameter<std::string>("trajectoryLayerEntrance")),
        _trajectoryLayerExitString(conf.getParameter<std::string>("trajectoryLayerExit")),
        _debug(conf.getUntrackedParameter<bool>("debug", false)) {
    // convert TrajectoryLayers info from string to enum
    _trajectoryLayerEntrance = reco::PFTrajectoryPoint::layerTypeFromString(_trajectoryLayerEntranceString);
    _trajectoryLayerExit = reco::PFTrajectoryPoint::layerTypeFromString(_trajectoryLayerExitString);
    // make sure the requested setting is supported
    assert((_trajectoryLayerEntrance == reco::PFTrajectoryPoint::HCALEntrance &&
            _trajectoryLayerExit == reco::PFTrajectoryPoint::HCALExit) ||
           (_trajectoryLayerEntrance == reco::PFTrajectoryPoint::HCALEntrance &&
            _trajectoryLayerExit == reco::PFTrajectoryPoint::Unknown) ||
           (_trajectoryLayerEntrance == reco::PFTrajectoryPoint::VFcalEntrance &&
            _trajectoryLayerExit == reco::PFTrajectoryPoint::Unknown));
    // flag if exit layer should be checked or not
    _checkExit = (_trajectoryLayerExit == reco::PFTrajectoryPoint::Unknown) ? false : true;
  }

  double testLink(const reco::PFBlockElement*, const reco::PFBlockElement*) const override;

private:
  bool _useKDTree;
  std::string _trajectoryLayerEntranceString;
  std::string _trajectoryLayerExitString;
  reco::PFTrajectoryPoint::LayerType _trajectoryLayerEntrance;
  reco::PFTrajectoryPoint::LayerType _trajectoryLayerExit;
  bool _debug;
  bool _checkExit;
};

DEFINE_EDM_PLUGIN(BlockElementLinkerFactory, TrackAndHCALLinker, "TrackAndHCALLinker");

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
  const reco::PFTrajectoryPoint& tkAtHCALEnt = trackref->extrapolatedPoint(_trajectoryLayerEntrance);
  // Check exit point
  double dHEta = 0.;
  double dHPhi = 0.;
  double dRHCALEx = 0.;
  if (_checkExit) {
    const reco::PFTrajectoryPoint& tkAtHCALEx = trackref->extrapolatedPoint(_trajectoryLayerExit);
    dHEta = (tkAtHCALEx.positionREP().Eta() - tkAtHCALEnt.positionREP().Eta());
    dHPhi = reco::deltaPhi(tkAtHCALEx.positionREP().Phi(), tkAtHCALEnt.positionREP().Phi());
    dRHCALEx = tkAtHCALEx.position().R();
  }
  if (_useKDTree && hcalelem->isMultilinksValide()) {  //KDTree Algo
    const reco::PFMultilinksType& multilinks = hcalelem->getMultilinks();
    const double tracketa = tkAtHCALEnt.positionREP().Eta();
    const double trackphi = tkAtHCALEnt.positionREP().Phi();

    // Check if the link Track/Hcal exist
    reco::PFMultilinksType::const_iterator mlit = multilinks.begin();
    for (; mlit != multilinks.end(); ++mlit)
      if ((mlit->first == trackphi) && (mlit->second == tracketa))
        break;

    // If the link exist, we fill dist and linktest.

    if (mlit != multilinks.end()) {
      // when _checkExit is false
      if (!_checkExit) {
        dist = LinkByRecHit::computeDist(hcalreppos.Eta(), hcalreppos.Phi(), tracketa, trackphi);
      }
      // when _checkExit is true
      else {
        //special case ! A looper  can exit the barrel inwards and hit the endcap
        //In this case calculate the distance based on the first crossing since
        //the looper will probably never make it to the endcap
        if (dRHCALEx < tkAtHCALEnt.position().R()) {
          dist = LinkByRecHit::computeDist(hcalreppos.Eta(), hcalreppos.Phi(), tracketa, trackphi);
          edm::LogWarning("TrackHCALLinker ")
              << "Special case of linking with track hitting HCAL and looping back in the tracker ";
        } else {
          dist = LinkByRecHit::computeDist(
              hcalreppos.Eta(), hcalreppos.Phi(), tracketa + 0.1 * dHEta, trackphi + 0.1 * dHPhi);
        }
      }  // _checkExit
    }    // multilinks

  } else {  // Old algorithm
    if (tkAtHCALEnt.isValid())
      dist = LinkByRecHit::testTrackAndClusterByRecHit(*trackref, *clusterref, false, _debug);
  }
  return dist;
}
