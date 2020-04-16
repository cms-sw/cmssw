#ifndef PFProducer_PFEGammaAlgo_H
#define PFProducer_PFEGammaAlgo_H

//
// Rewrite for GED integration:  Lindsey Gray (FNAL): lagray@fnal.gov
//
// Original Authors: Fabian Stoeckli: fabian.stoeckli@cern.ch
//                   Nicholas Wardle: nckw@cern.ch
//                   Rishi Patel: rpatel@cern.ch
//                   Josh Bendavid : Josh.Bendavid@cern.ch
//

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtraFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtraFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "CondFormats/ESObjects/interface/ESEEIntercalibConstants.h"
#include "CondFormats/ESObjects/interface/ESChannelStatus.h"

#include <iostream>
#include <TH2D.h>

#include <list>
#include <forward_list>
#include <unordered_map>

#include "RecoParticleFlow/PFProducer/interface/FlaggedPtr.h"
#include "RecoParticleFlow/PFProducer/interface/CommutativePairs.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/MVAUtils/interface/GBRForestTools.h"

#include <memory>

class PFEnergyCalibration;

class PFEGammaAlgo {
public:
  typedef reco::PFCluster::EEtoPSAssociation EEtoPSAssociation;
  typedef reco::PFBlockElementSuperCluster PFSCElement;
  typedef reco::PFBlockElementBrem PFBremElement;
  typedef reco::PFBlockElementGsfTrack PFGSFElement;
  typedef reco::PFBlockElementTrack PFKFElement;
  typedef reco::PFBlockElementCluster PFClusterElement;
  typedef std::unordered_map<const PFKFElement*, float> KFValMap;

  using ClusterMap = std::unordered_map<PFClusterElement const*, std::vector<PFClusterElement const*>>;

  class GBRForests {
  public:
    GBRForests(const edm::ParameterSet& conf)
        : ele_(createGBRForest(conf.getParameter<edm::FileInPath>("pf_electronID_mvaWeightFile"))),
          singleLeg_(createGBRForest(conf.getParameter<edm::FileInPath>("pf_convID_mvaWeightFile"))) {}

    const std::unique_ptr<const GBRForest> ele_;
    const std::unique_ptr<const GBRForest> singleLeg_;
  };

  struct ProtoEGObject {
    reco::PFBlockRef parentBlock;
    const PFSCElement* parentSC = nullptr;  // if ECAL driven
    reco::ElectronSeedRef electronSeed;     // if there is one
    // this is a mutable list of clusters
    // if ECAL driven we take the PF SC and refine it
    // if Tracker driven we add things to it as we discover more valid clusters
    std::vector<FlaggedPtr<const PFClusterElement>> ecalclusters;
    ClusterMap ecal2ps;
    // associations to tracks of various sorts
    std::vector<PFGSFElement const*> primaryGSFs;
    std::vector<PFKFElement const*> primaryKFs;
    std::vector<PFBremElement const*> brems;  // these are tangent based brems
    // for manual brem recovery
    std::vector<PFGSFElement const*> secondaryGSFs;
    std::vector<PFKFElement const*> secondaryKFs;
    KFValMap singleLegConversionMvaMap;
    // for track-HCAL cluster linking
    std::vector<PFClusterElement const*> hcalClusters;
    CommutativePairs<const reco::PFBlockElement*> localMap;
    // cluster closest to the gsf track(s), primary kf if none for gsf
    // last brem tangent cluster if neither of those work
    std::vector<const PFClusterElement*> electronClusters;
    int firstBrem, lateBrem, nBremsWithClusters;
  };

  struct PFEGConfigInfo {
    double mvaEleCut;
    bool applyCrackCorrections;
    bool produceEGCandsWithNoSuperCluster;
    double mvaConvCut;
  };

  struct EgammaObjects {
    reco::PFCandidateCollection candidates;
    reco::PFCandidateEGammaExtraCollection candidateExtras;
    reco::SuperClusterCollection refinedSuperClusters;
  };

  //constructor
  PFEGammaAlgo(const PFEGConfigInfo&,
               GBRForests const& gbrForests,
               EEtoPSAssociation const& eetops,
               ESEEIntercalibConstants const& esEEInterCalib,
               ESChannelStatus const& channelStatus,
               reco::Vertex const& primaryVertex);

  // this runs the functions below
  EgammaObjects operator()(const reco::PFBlockRef& block);

private:
  GBRForests const& gbrForests_;

  PFEnergyCalibration thePFEnergyCalibration_;

  // ------ rewritten basic processing pieces and cleaning algorithms

  // useful pre-cached mappings:
  // hopefully we get an enum that lets us just make an array in the future
  reco::PFCluster::EEtoPSAssociation const& eetops_;
  reco::PFBlockRef _currentblock;
  reco::PFBlock::LinkData _currentlinks;
  // keep a map of pf indices to the splayed block for convenience
  // sadly we're mashing together two ways of thinking about the block
  std::vector<std::vector<FlaggedPtr<const reco::PFBlockElement>>> _splayedblock;

  // pre-cleaning for the splayed block
  bool isMuon(const reco::PFBlockElement&);
  // pre-processing of ECAL clusters near non-primary KF tracks
  void removeOrLinkECALClustersToKFTracks();

  // functions:

  // build proto eg object using all available unflagged resources in block.
  // this will be kind of like the old 'SetLinks' but with simplified and
  // maximally inclusive logic that builds a list of 'refinable' objects
  // that we will perform operations on to clean/remove as needed
  void initializeProtoCands(std::list<ProtoEGObject>&);

  // turn a supercluster into a map of ECAL cluster elements
  // related to PS cluster elements
  bool unwrapSuperCluster(const reco::PFBlockElementSuperCluster*,
                          std::vector<FlaggedPtr<const PFClusterElement>>&,
                          ClusterMap&);

  int attachPSClusters(const PFClusterElement*, ClusterMap::mapped_type&);

  void dumpCurrentRefinableObjects() const;

  // wax on

  // the key merging operation, done after building up links
  void mergeROsByAnyLink(std::list<ProtoEGObject>&);

  // refining steps you can do with tracks
  void linkRefinableObjectGSFTracksToKFs(ProtoEGObject&);
  void linkRefinableObjectPrimaryKFsToSecondaryKFs(ProtoEGObject&);
  void linkRefinableObjectPrimaryGSFTrackToECAL(ProtoEGObject&);
  void linkRefinableObjectPrimaryGSFTrackToHCAL(ProtoEGObject&);
  void linkRefinableObjectKFTracksToECAL(ProtoEGObject&);
  void linkRefinableObjectBremTangentsToECAL(ProtoEGObject&);
  // WARNING! this should be ONLY used after doing the ECAL->track
  // reverse lookup after the primary linking!
  void linkRefinableObjectConvSecondaryKFsToSecondaryKFs(ProtoEGObject&);
  void linkRefinableObjectSecondaryKFsToECAL(ProtoEGObject&);
  // helper function for above
  void linkKFTrackToECAL(PFKFElement const*, ProtoEGObject&);

  // refining steps doing the ECAL -> track piece
  // this is the factorization of the old PF photon algo stuff
  // which through arcane means I came to understand was conversion matching
  void linkRefinableObjectECALToSingleLegConv(ProtoEGObject&);

  // wax off

  // refining steps remove things from the built-up objects
  // original bits were for removing bad KF tracks
  // new (experimental) piece to remove clusters associated to these tracks
  // behavior determined by bools passed to unlink_KFandECALMatchedToHCAL
  void unlinkRefinableObjectKFandECALWithBadEoverP(ProtoEGObject&);
  void unlinkRefinableObjectKFandECALMatchedToHCAL(ProtoEGObject&,
                                                   bool removeFreeECAL = false,
                                                   bool removeSCECAL = false);

  // things for building the final candidate and refined SC collections
  EgammaObjects fillPFCandidates(const std::list<ProtoEGObject>&);
  reco::SuperCluster buildRefinedSuperCluster(const ProtoEGObject&);

  // helper functions for that

  float calculateEleMVA(const ProtoEGObject&, reco::PFCandidateEGammaExtra&) const;
  void fillExtraInfo(const ProtoEGObject&, reco::PFCandidateEGammaExtra&);

  // ------ end of new stuff

  bool isPrimaryTrack(const reco::PFBlockElementTrack& KfEl, const reco::PFBlockElementGsfTrack& GsfEl);

  PFEGConfigInfo const& cfg_;
  reco::Vertex const& primaryVertex_;

  ESChannelStatus const& channelStatus_;

  float evaluateSingleLegMVA(const reco::PFBlockRef& blockref, const reco::Vertex& primaryVtx, unsigned int trackIndex);
};

#endif
