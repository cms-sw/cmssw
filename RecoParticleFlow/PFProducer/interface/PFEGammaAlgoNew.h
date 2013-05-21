#ifndef PFProducer_PFEGammaAlgoNew_H
#define PFProducer_PFEGammaAlgoNew_H

//
// Original Authors: Fabian Stoeckli: fabian.stoeckli@cern.ch
//                   Nicholas Wardle: nckw@cern.ch
//                   Rishi Patel rpatel@cern.ch
// Rewrite for GED integration: Josh Bendavid (CERN), Lindsey Gray (FNAL)
//

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
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
#include "TMVA/Reader.h"
#include <iostream>
#include <TH2D.h>

#include <list>
#include <unordered_map>

class PFSCEnergyCalibration;
class PFEnergyCalibration;

class PFEGammaAlgoNew {
 public:
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
  typedef std::unordered_multimap<const reco::PFBlockElement*,
    const reco::PFBlockElement*> ElementMap;
  typedef std::unordered_map<const PFGSFElement*, 
    std::vector<PFKFFlaggedElement> > GSFToTrackMap;
  typedef std::unordered_map<const PFClusterElement*, 
    std::vector<PFClusterFlaggedElement> > ClusterMap;  

  struct ProtoEGObject {
    ProtoEGObject() : parentSC(NULL) {}
    reco::PFBlockRef parentBlock;
    const PFSCElement* parentSC; // if ECAL driven
    reco::ElectronSeedRef electronSeed; // if there is one
    // this is a mutable list of clusters
    // if ECAL driven we take the PF SC and refine it
    // if Tracker driven we add things to it as we discover more valid clusters
    std::list<PFClusterFlaggedElement> ecalclusters;
    ClusterMap ecal2ps;
    // associations to tracks of various sorts
    std::vector<PFGSFFlaggedElement> primaryGSFs; 
    GSFToTrackMap boundKFTracks;
    std::vector<PFKFFlaggedElement> primaryKFs;
    std::vector<PFBremFlaggedElement> brems; // these are tangent based brems
    // for manual brem recovery 
    std::vector<PFGSFFlaggedElement> secondaryGSFs;
    std::vector<PFKFFlaggedElement> secondaryKFs;    
    // for track-HCAL cluster linking
    std::vector<PFClusterFlaggedElement> hcalClusters;
    ElementMap localMap;
  };  
  
  //constructor
  PFEGammaAlgoNew(const double mvaEleCut,
	       std::string  mvaWeightFileEleID,
	       const std::shared_ptr<PFSCEnergyCalibration>& thePFSCEnergyCalibration,
	       const std::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
	       bool applyCrackCorrections,
	       bool usePFSCEleCalib,
	       bool useEGElectrons,
	       bool useEGammaSupercluster,
	       double sumEtEcalIsoForEgammaSC_barrel,
	       double sumEtEcalIsoForEgammaSC_endcap,
	       double coneEcalIsoForEgammaSC,
	       double sumPtTrackIsoForEgammaSC_barrel,
	       double sumPtTrackIsoForEgammaSC_endcap,
	       unsigned int nTrackIsoForEgammaSC,
	       double coneTrackIsoForEgammaSC,
	       std::string mvaweightfile,  
	       double mvaConvCut, 
	       bool useReg, 
	       std::string X0_Map,
	       const reco::Vertex& primary,
               double sumPtTrackIsoForPhoton,
               double sumPtTrackIsoSlopeForPhoton); 

  //destructor
  ~PFEGammaAlgoNew(){delete tmvaReaderEle_; delete tmvaReader_;   };

  void setGBRForest(const GBRForest *LCorrForest,
		    const GBRForest *GCorrForest,
		    const GBRForest *ResForest
		    )
  {
    ReaderLC_=LCorrForest;
    ReaderGC_=GCorrForest;
    ReaderRes_=ResForest;
  }  
  
  void setGBRForest(
		    const GBRForest *LCorrForestEB,
		    const GBRForest *LCorrForestEE,
		    const GBRForest *GCorrForestBarrel,
		    const GBRForest *GCorrForestEndcapHr9,
		    const GBRForest *GCorrForestEndcapLr9,
		    const GBRForest *PFEcalResolution
		    )
  {
    ReaderLCEB_=LCorrForestEB;
    ReaderLCEE_=LCorrForestEE;
    ReaderGCEB_=GCorrForestBarrel;
    ReaderGCEEhR9_=GCorrForestEndcapHr9;
    ReaderGCEElR9_=GCorrForestEndcapLr9;
    ReaderRes_=PFEcalResolution;
  }  
  void setnPU(int nVtx){
    nVtx_=nVtx;
  }
  void setPhotonPrimaryVtx(const reco::Vertex& primary){
    primaryVertex_ = & primary;
  }

  void RunPFEG(const reco::PFBlockRef&  blockRef,
	       std::vector< bool >& active
	       );

  //check candidate validity
  bool isEGValidCandidate(const reco::PFBlockRef&  blockRef,
			      std::vector< bool >&  active
			      //      std::auto_ptr< reco::PFCandidateCollection > &pfElectronCandidates_  
			      ){
    RunPFEG(blockRef,active);
    return (egCandidate_.size()>0);
  };
  
  //get PFCandidate collection
  const reco::PFCandidateCollection& getCandidates() {return egCandidate_;}

  //get the PFCandidateExtra (for all candidates)
  const reco::PFCandidateEGammaExtraCollection& getEGExtra() {return egExtra_;}
  
  //get electron PFCandidate
  
  
private: 
  

  enum verbosityLevel {
    Silent,
    Summary,
    Chatty
  };

  // ------ rewritten basic processing pieces and cleaning algorithms
  
  // usefule pre-cached mappings:
  // hopefully we get an enum that lets us just make an array in the future
  reco::PFBlockRef _currentblock;
  reco::PFBlock::LinkData _currentlinks;  
  // keep a map of pf indices to the splayed block for convenience
  // sadly we're mashing together two ways of thinking about the block
  std::vector<std::vector<PFFlaggedElement> > _splayedblock; 
  ElementMap _recoveredlinks;

  // pre-cleaning for the splayed block
  bool isAMuon(const reco::PFBlockElement&);
  // pre-processing of ECAL clusters near non-primary KF tracks
  void removeOrLinkECALClustersToKFTracks();

  // candidate collections:
  // this starts off as an inclusive list of prototype objects built from 
  // supercluster/ecal-driven seeds and tracker driven seeds in a block
  // it is then refined through by various cleanings, determining the energy 
  // flow.
  // use list for constant-time removals
  std::list<ProtoEGObject> _refinableObjects;
  // final list of fully refined objects in this block
  reco::PFCandidateCollection _finalCandidates;

  // functions:
  // this runs the above functions
  void buildAndRefineEGObjects(const reco::PFBlockRef& block);

  // build proto eg object using all available unflagged resources in block.
  // this will be kind of like the old 'SetLinks' but with simplified and 
  // maximally inclusive logic that builds a list of 'refinable' objects
  // that we will perform operations on to clean/remove as needed
  void initializeProtoCands(std::list<ProtoEGObject>&);

  // turn a supercluster into a map of ECAL cluster elements 
  // related to PS cluster elements
  void unwrapSuperCluster(const reco::PFBlockElementSuperCluster*,
			  std::list<PFClusterFlaggedElement>&,
			  ClusterMap&);    
  
  int attachPSClusters(const PFClusterElement*,
		       ClusterMap::mapped_type&);    

  
  void dumpCurrentRefinableObjects() const;
  
  // the key merging operation, done after building up links
  void mergeROsByAnyLink(std::list<ProtoEGObject>&);

  // refining steps you can do with KF tracks
  void linkRefinableObjectGSFTracksToKFs(std::list<ProtoEGObject>&);
  void linkRefinableObjectPrimaryKFsToSecondaryKFs(std::list<ProtoEGObject>&);
  void linkRefinableObjectKFTracksToECAL(std::list<ProtoEGObject>&);
  // helper function for above
  void linkKFTrackToECAL(const PFKFFlaggedElement&, ProtoEGObject&);

  // ------ end of new stuff 


  // NB:
  // associatedToGsf = SC linked to primary GSF track by matched cluster
  // associatedToBrem = SC associated to secondary GSF track by matched cluster
  //                  - care is taken to make sure there is not a closer
  //                  - primary gsf track.
  // associatedToEcal = HCAL/GSF/Brems/Tracks associated to ECAL pieces

  bool SetLinks(const reco::PFBlockRef&  blockRef,
		AsscMap& associatedToGsf_,
		AsscMap& associatedToBrems_,
		AsscMap& associatedToEcal_,
		std::vector<bool>& active,
		const reco::Vertex & primaryVertex);
  
  unsigned int whichTrackAlgo(const reco::TrackRef& trackRef);

  bool isPrimaryTrack(const reco::PFBlockElementTrack& KfEl,
		      const reco::PFBlockElementGsfTrack& GsfEl);  
  
  void AddElectronElements(unsigned int gsf_index,
			   std::vector<unsigned int> &elemsToLock,
			   const reco::PFBlockRef&  blockRef,
			   AsscMap& associatedToGsf_,
			   AsscMap& associatedToBrems_,
			   AsscMap& associatedToEcal_);
  

  bool AddElectronCandidate(unsigned int gsf_index,
			    reco::SuperClusterRef scref,
			    std::vector<unsigned int> &elemsToLock,
			    const reco::PFBlockRef&  blockRef,
			    AsscMap& associatedToGsf_,
			    AsscMap& associatedToBrems_,
			    AsscMap& associatedToEcal_,
			    std::vector<bool>& active); 
  
 //Data members from PFElectronAlgo
//   std::vector<reco::PFCandidate> elCandidate_;
//   std::vector<reco::PFCandidate> allElCandidate_;
  //std::map<unsigned int,std::vector<reco::PFCandidate> > electronConstituents_;
  //std::vector<double> BDToutput_;
  //std::vector<reco::PFCandidateElectronExtra > electronExtra_;
  std::vector<bool> lockExtraKf_;
  std::vector<bool> GsfTrackSingleEcal_;
  std::vector< std::pair <unsigned int, unsigned int> > fifthStepKfTrack_;
  std::vector< std::pair <unsigned int, unsigned int> > convGsfTrack_;

  
  TMVA::Reader    *tmvaReaderEle_;
  double mvaEleCut_;
  std::shared_ptr<PFSCEnergyCalibration> thePFSCEnergyCalibration_; 
  std::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_; 
  bool applyCrackCorrections_;
  bool usePFSCEleCalib_;
  bool useEGElectrons_;
  bool useEGammaSupercluster_;
  double sumEtEcalIsoForEgammaSC_barrel_;
  double sumEtEcalIsoForEgammaSC_endcap_;
  double coneEcalIsoForEgammaSC_;
  double sumPtTrackIsoForEgammaSC_barrel_;
  double sumPtTrackIsoForEgammaSC_endcap_;
  unsigned int nTrackIsoForEgammaSC_;
  double coneTrackIsoForEgammaSC_;

  const char  *mvaWeightFile_;

  // New BDT observables
  // Normalization 
  float lnPt_gsf,Eta_gsf;
  
  // Pure Tracking observ.
  float dPtOverPt_gsf,chi2_gsf,DPtOverPt_gsf,
    chi2_kf,DPtOverPt_kf;
  //  int nhit_gsf,nhit_kf;
  float nhit_gsf,nhit_kf;
  
  // Tracker-Ecal observ. 
  float EtotPinMode,EGsfPoutMode,EtotBremPinPoutMode;
  float DEtaGsfEcalClust;
  float SigmaEtaEta; 
  //int lateBrem,firstBrem,earlyBrem;
  float lateBrem,firstBrem,earlyBrem;
  float HOverHE,HOverPin;

  bool isvalid_;

  //const std::vector<reco::GsfElectron> * theGsfElectrons_;
  //end of data members from PFElectronAlgo
  
  
  //bool isvalid_;                               // is set to TRUE when a valid PhotonCandidate is found in a PFBlock
  verbosityLevel  verbosityLevel_;            /* Verbosity Level: 
						  ...............  0: Say nothing at all
						  ...............  1: Print summary about found PhotonCadidates only
						  ...............  2: Chatty mode
                                              */ 
  //FOR SINGLE LEG MVA:					      
  double MVACUT;
  bool useReg_;
  const reco::Vertex  *  primaryVertex_;
  TMVA::Reader *tmvaReader_;
  const GBRForest *ReaderLC_;
  const GBRForest *ReaderGC_;
  const GBRForest *ReaderRes_;
  
  const GBRForest *ReaderLCEB_;
  const GBRForest *ReaderLCEE_;
  const GBRForest *ReaderGCEB_;
  const GBRForest *ReaderGCEEhR9_;
  const GBRForest *ReaderGCEElR9_;
  
//  boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_;
  double sumPtTrackIsoForPhoton_;
  double sumPtTrackIsoSlopeForPhoton_;
  std::vector<int>match_ind;
  //std::auto_ptr< reco::PFCandidateCollection > permElectronCandidates_;

  std::vector< reco::PFCandidate >permElectronCandidates_;
  float nlost, nlayers;
  float chi2, STIP, del_phi,HoverPt, EoverPt, track_pt;
  double mvaValue;
    //for Cluster Shape Calculations:
  float e5x5Map[5][5];
  
  //For Local Containment Corrections:
  float CrysPhi_, CrysEta_,  VtxZ_, ClusPhi_, ClusEta_, 
    ClusR9_, Clus5x5ratio_,  PFCrysEtaCrack_, logPFClusE_, e3x3_;
  int CrysIPhi_, CrysIEta_;
  float CrysX_, CrysY_;
  float EB;
  //Cluster Shapes:
  float eSeed_, e1x3_,e3x1_, e1x5_, e2x5Top_,  e2x5Bottom_, e2x5Left_,  e2x5Right_ ;
  float etop_, ebottom_, eleft_, eright_;
  float e2x5Max_;
  //For Global Corrections:
  float PFPhoEta_, PFPhoPhi_, PFPhoR9_, PFPhoR9Corr_, SCPhiWidth_, SCEtaWidth_, PFPhoEt_, RConv_, PFPhoEtCorr_, PFPhoE_, PFPhoECorr_, MustE_, E3x3_;
  float dEta_, dPhi_, LowClusE_, RMSAll_, RMSMust_, nPFClus_;
  float TotPS1_, TotPS2_;
  float nVtx_;
  //for Material Map
  TH2D* X0_sum;
  TH2D* X0_inner;
  TH2D* X0_middle;
  TH2D* X0_outer;
  float x0inner_, x0middle_, x0outer_;
  //for PileUP
  float excluded_, Mustache_EtRatio_, Mustache_Et_out_;
  
  std::vector<unsigned int> AddFromElectron_;  
  
  reco::PFCandidateCollection egCandidate_;
//   std::vector<reco::CaloCluser> ebeeCluster_;
//   std::vector<reco::PreshowerCluser> esCluster_;
//   std::vector<reco::SuperCluser> sCluster_;
  reco::PFCandidateEGammaExtraCollection egExtra_;  

  bool EvaluateSingleLegMVA(const reco::PFBlockRef& blockref, 
			    const reco::Vertex& primaryvtx, 
			    unsigned int track_index);
  
  double ClustersPhiRMS(std::vector<reco::CaloCluster>PFClusters, 
			float PFPhoPhi);
  float EvaluateLCorrMVA(reco::PFClusterRef clusterRef );
  float EvaluateGCorrMVA(reco::PFCandidate, 
			 std::vector<reco::CaloCluster>PFClusters);
  float EvaluateResMVA(reco::PFCandidate,
		       std::vector<reco::CaloCluster>PFClusters );
  std::vector<int> getPFMustacheClus(int nClust, 
				     std::vector<float>& ClustEt, 
				     std::vector<float>& ClustEta, 
				     std::vector<float>& ClustPhi);
  void EarlyConversion(
		       //std::auto_ptr< reco::PFCandidateCollection > 
		       //&pfElectronCandidates_,
		       std::vector<reco::PFCandidate>& 
		       tempElectronCandidates,
		       const reco::PFBlockElementSuperCluster* sc
		       );
};

#endif
