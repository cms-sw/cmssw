#ifndef PFProducer_PFEGammaAlgo_H
#define PFProducer_PFEGammaAlgo_H

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtraFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "TMVA/Reader.h"
#include <iostream>
#include <TH2D.h>

class PFSCEnergyCalibration;
class PFEnergyCalibration;

namespace reco {
  class PFCandidate;
  class PFCandidateCollectioon;
}

class PFEGammaAlgo {
 public:
  
  //constructor
  PFEGammaAlgo(const double mvaEleCut,
	       std::string  mvaWeightFileEleID,
	       const boost::shared_ptr<PFSCEnergyCalibration>& thePFSCEnergyCalibration,
	       const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
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
               double sumPtTrackIsoSlopeForPhoton
); 

  //destructor
  ~PFEGammaAlgo(){delete tmvaReaderEle_; delete tmvaReader_;   };

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
  //check candidate validity
  bool isEGValidCandidate(const reco::PFBlockRef&  blockRef,
			      std::vector< bool >&  active
			      //      std::auto_ptr< reco::PFCandidateCollection > &pfElectronCandidates_  
			      ){
    RunPFEG(blockRef,active);
    return (egCandidate_.size()>0);
  };
  
  //get PFCandidate collection
  const std::vector<reco::PFCandidate>& getCandidates() {return egCandidate_;};

  //get the PFCandidateExtra (for all candidates)
  const std::vector< reco::PFCandidateEGammaExtra>& getEGExtra() {return egExtra_;};  
  
  //get electron PFCandidate
  
  
private: 
  typedef  std::map< unsigned int, std::vector<unsigned int> >  AssMap;

  enum verbosityLevel {
    Silent,
    Summary,
    Chatty
  };
  

  bool SetLinks(const reco::PFBlockRef&  blockRef,
		AssMap& associatedToGsf_,
		AssMap& associatedToBrems_,
		AssMap& associatedToEcal_,
		std::vector<bool>& active,
		const reco::Vertex & primaryVertex);
  
  unsigned int whichTrackAlgo(const reco::TrackRef& trackRef);

  bool isPrimaryTrack(const reco::PFBlockElementTrack& KfEl,
		      const reco::PFBlockElementGsfTrack& GsfEl);  
  
  void AddElectronElements(unsigned int gsf_index,
			             std::vector<unsigned int> &elemsToLock,
				     const reco::PFBlockRef&  blockRef,
				     AssMap& associatedToGsf_,
				     AssMap& associatedToBrems_,
				     AssMap& associatedToEcal_);
  

  bool AddElectronCandidate(unsigned int gsf_index,
			    reco::SuperClusterRef scref,
					 std::vector<unsigned int> &elemsToLock,
					 const reco::PFBlockRef&  blockRef,
					 AssMap& associatedToGsf_,
					 AssMap& associatedToBrems_,
					 AssMap& associatedToEcal_,
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
  boost::shared_ptr<PFSCEnergyCalibration> thePFSCEnergyCalibration_; 
  boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_; 
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
  
  std::vector<reco::PFCandidate> egCandidate_;
//   std::vector<reco::CaloCluser> ebeeCluster_;
//   std::vector<reco::PreshowerCluser> esCluster_;
//   std::vector<reco::SuperCluser> sCluster_;
  std::vector<reco::PFCandidateEGammaExtra> egExtra_;

   
  
  
  
  void RunPFEG(const reco::PFBlockRef&  blockRef,
		   std::vector< bool >& active
		   );

  bool EvaluateSingleLegMVA(const reco::PFBlockRef& blockref, 
			    const reco::Vertex& primaryvtx, 
			    unsigned int track_index);
  
  double ClustersPhiRMS(std::vector<reco::CaloCluster>PFClusters, float PFPhoPhi);
  float EvaluateLCorrMVA(reco::PFClusterRef clusterRef );
  float EvaluateGCorrMVA(reco::PFCandidate, std::vector<reco::CaloCluster>PFClusters);
  float EvaluateResMVA(reco::PFCandidate,std::vector<reco::CaloCluster>PFClusters );
  std::vector<int> getPFMustacheClus(int nClust, std::vector<float>& ClustEt, std::vector<float>& ClustEta, std::vector<float>& ClustPhi);
  void EarlyConversion(
		       //std::auto_ptr< reco::PFCandidateCollection > 
		       //&pfElectronCandidates_,
		       std::vector<reco::PFCandidate>& 
		       tempElectronCandidates,
		       const reco::PFBlockElementSuperCluster* sc
		       );
};

#endif
