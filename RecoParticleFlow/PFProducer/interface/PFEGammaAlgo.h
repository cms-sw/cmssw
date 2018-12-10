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

#include "RecoParticleFlow/PFProducer/interface/PFEGammaHeavyObjectCache.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

class PFSCEnergyCalibration;
class PFEnergyCalibration;

class PFEGammaAlgo {
 public:
  typedef reco::PFCluster::EEtoPSAssociation EEtoPSAssociation;
  typedef reco::PFBlockElementSuperCluster PFSCElement;
  typedef reco::PFBlockElementBrem PFBremElement;
  typedef reco::PFBlockElementGsfTrack PFGSFElement;
  typedef reco::PFBlockElementTrack PFKFElement;
  typedef reco::PFBlockElementCluster PFClusterElement;
  typedef std::pair<const reco::PFBlockElement*,bool> PFFlaggedElement;
  typedef std::pair<const PFSCElement*,bool> PFSCFlaggedElement;
  typedef std::pair<const PFBremElement*,bool> PFBremFlaggedElement;
  typedef std::pair<const PFGSFElement*,bool> PFGSFFlaggedElement;
  typedef std::pair<const PFKFElement*,bool> PFKFFlaggedElement;
  typedef std::pair<const PFClusterElement*,bool> PFClusterFlaggedElement;
  typedef std::unordered_map<unsigned int, std::vector<unsigned int> > AsscMap;
  typedef std::vector<std::pair<const reco::PFBlockElement*,
    const reco::PFBlockElement*> > ElementMap;
  typedef std::unordered_map<const PFGSFElement*, 
    std::vector<PFKFFlaggedElement> > GSFToTrackMap;
  typedef std::unordered_map<const PFClusterElement*, 
    std::vector<PFClusterFlaggedElement> > ClusterMap;  
  typedef std::unordered_map<const PFKFElement*, 
    float > KFValMap;  
    
  struct ProtoEGObject {
    ProtoEGObject() : parentSC(nullptr) {}
    reco::PFBlockRef parentBlock;
    const PFSCElement* parentSC; // if ECAL driven
    reco::ElectronSeedRef electronSeed; // if there is one
    // this is a mutable list of clusters
    // if ECAL driven we take the PF SC and refine it
    // if Tracker driven we add things to it as we discover more valid clusters
    std::vector<PFClusterFlaggedElement> ecalclusters;
    ClusterMap ecal2ps;
    // associations to tracks of various sorts
    std::vector<PFGSFFlaggedElement> primaryGSFs; 
    GSFToTrackMap boundKFTracks;
    std::vector<PFKFFlaggedElement> primaryKFs;
    std::vector<PFBremFlaggedElement> brems; // these are tangent based brems
    // for manual brem recovery 
    std::vector<PFGSFFlaggedElement> secondaryGSFs;
    std::vector<PFKFFlaggedElement> secondaryKFs;    
    KFValMap singleLegConversionMvaMap;
    // for track-HCAL cluster linking
    std::vector<PFClusterFlaggedElement> hcalClusters;
    ElementMap localMap;
    // cluster closest to the gsf track(s), primary kf if none for gsf
    // last brem tangent cluster if neither of those work
    std::vector<const PFClusterElement*> electronClusters; 
    int firstBrem, lateBrem, nBremsWithClusters;
  };  
  
  struct PFEGConfigInfo {
    double mvaEleCut;
    std::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration;
    bool applyCrackCorrections;
    bool produceEGCandsWithNoSuperCluster;
    double mvaConvCut;
    const reco::Vertex* primaryVtx;
  };

  //constructor
  PFEGammaAlgo(const PFEGConfigInfo&);
  //destructor
  ~PFEGammaAlgo(){ };

  void setEEtoPSAssociation(const edm::Handle<EEtoPSAssociation>& eetops) {
    eetops_ = eetops;
  }

  void setAlphaGamma_ESplanes_fromDB(const ESEEIntercalibConstants* esEEInterCalib){
    cfg_.thePFEnergyCalibration->initAlphaGamma_ESplanes_fromDB(esEEInterCalib);
  }

  void setESChannelStatus(const ESChannelStatus* channelStatus){
    channelStatus_ = channelStatus;
  }

  void setPhotonPrimaryVtx(const reco::Vertex& primary){
    cfg_.primaryVtx = & primary;
  }

  void RunPFEG(const pfEGHelpers::HeavyObjectCache* hoc,
               const reco::PFBlockRef&  blockRef);
  
  //get PFCandidate collection
  reco::PFCandidateCollection& getCandidates() {return outcands_;}

  //get the PFCandidateExtra (for all candidates)
  reco::PFCandidateEGammaExtraCollection& getEGExtra() {return outcandsextra_;}
  
  //get refined SCs
  reco::SuperClusterCollection& getRefinedSCs() {return refinedscs_;}
  
private: 
  

  // ------ rewritten basic processing pieces and cleaning algorithms
  // the output collections
  reco::PFCandidateCollection outcands_;
  reco::PFCandidateEGammaExtraCollection outcandsextra_;
  reco::SuperClusterCollection refinedscs_;

  // useful pre-cached mappings:
  // hopefully we get an enum that lets us just make an array in the future
  edm::Handle<reco::PFCluster::EEtoPSAssociation> eetops_;
  reco::PFBlockRef _currentblock;
  reco::PFBlock::LinkData _currentlinks;  
  // keep a map of pf indices to the splayed block for convenience
  // sadly we're mashing together two ways of thinking about the block
  std::vector<std::vector<PFFlaggedElement> > _splayedblock; 

  // pre-cleaning for the splayed block
  bool isMuon(const reco::PFBlockElement&);
  // pre-processing of ECAL clusters near non-primary KF tracks
  void removeOrLinkECALClustersToKFTracks();

  // candidate collections:
  // this starts off as an inclusive list of prototype objects built from 
  // supercluster/ecal-driven seeds and tracker driven seeds in a block
  // it is then refined through by various cleanings, determining the energy 
  // flow.
  // use list for constant-time removals
  std::list<ProtoEGObject> _refinableObjects;

  // functions:
  // this runs the functions below
  void buildAndRefineEGObjects(const pfEGHelpers::HeavyObjectCache* hoc,
                               const reco::PFBlockRef& block);

  // build proto eg object using all available unflagged resources in block.
  // this will be kind of like the old 'SetLinks' but with simplified and 
  // maximally inclusive logic that builds a list of 'refinable' objects
  // that we will perform operations on to clean/remove as needed
  void initializeProtoCands(std::list<ProtoEGObject>&);

  // turn a supercluster into a map of ECAL cluster elements 
  // related to PS cluster elements
  bool unwrapSuperCluster(const reco::PFBlockElementSuperCluster*,
			  std::vector<PFClusterFlaggedElement>&,
			  ClusterMap&);    
  
  int attachPSClusters(const PFClusterElement*,
		       ClusterMap::mapped_type&);  

  
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
  void linkKFTrackToECAL(const PFKFFlaggedElement&, ProtoEGObject&);

  // refining steps doing the ECAL -> track piece
  // this is the factorization of the old PF photon algo stuff
  // which through arcane means I came to understand was conversion matching  
  void linkRefinableObjectECALToSingleLegConv(const pfEGHelpers::HeavyObjectCache* hoc,
                                              ProtoEGObject&);

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
  void fillPFCandidates(const pfEGHelpers::HeavyObjectCache* hoc,
                        const std::list<ProtoEGObject>&, 
			reco::PFCandidateCollection&,
			reco::PFCandidateEGammaExtraCollection&);
  reco::SuperCluster buildRefinedSuperCluster(const ProtoEGObject&);
  
  // helper functions for that

  float calculateEleMVA(const pfEGHelpers::HeavyObjectCache* hoc,
                        const ProtoEGObject&,
                        reco::PFCandidateEGammaExtra&) const;
  void fillExtraInfo(const ProtoEGObject&,
		       reco::PFCandidateEGammaExtra&);
  
  // ------ end of new stuff 
  
  


  bool isPrimaryTrack(const reco::PFBlockElementTrack& KfEl,
		      const reco::PFBlockElementGsfTrack& GsfEl);  
  
 
  //std::vector<double> BDToutput_;
  //std::vector<reco::PFCandidateElectronExtra > electronExtra_;
  std::vector<bool> lockExtraKf_;
  std::vector<bool> GsfTrackSingleEcal_;
  std::vector< std::pair <unsigned int, unsigned int> > fifthStepKfTrack_;
  std::vector< std::pair <unsigned int, unsigned int> > convGsfTrack_;

  PFEGConfigInfo cfg_;

  const char  *mvaWeightFile_;

  const ESChannelStatus* channelStatus_;
  
  float evaluateSingleLegMVA(const pfEGHelpers::HeavyObjectCache* hoc,
                             const reco::PFBlockRef& blockref, 
                             const reco::Vertex& primaryVtx,
                             unsigned int trackIndex);
};

#endif
