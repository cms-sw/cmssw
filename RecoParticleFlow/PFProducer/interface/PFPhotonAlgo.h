#ifndef PFProducer_PFPhotonAlgo_H
#define PFProducer_PFPhotonAlgo_H

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

#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "TMVA/Reader.h"
#include <iostream>
#include <TH2D.h>

class PFEnergyCalibration;

namespace reco {
  class PFCandidate;
  class PFCandidateCollectioon;
}

class PFPhotonAlgo {
 public:
  
  //constructor
  PFPhotonAlgo(std::string mvaweightfile,  
	       double mvaConvCut, 
	       bool useReg, 
	       std::string X0_Map,
	       const reco::Vertex& primary,
	       const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
               double sumPtTrackIsoForPhoton,
               double sumPtTrackIsoSlopeForPhoton
); 

  //destructor
  ~PFPhotonAlgo(){delete tmvaReader_;   };

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
  bool isPhotonValidCandidate(const reco::PFBlockRef&  blockRef,
			      std::vector< bool >&  active,
			      std::unique_ptr< reco::PFCandidateCollection > &pfPhotonCandidates,
			      std::vector<reco::PFCandidatePhotonExtra>& pfPhotonExtraCandidates,
			      std::vector<reco::PFCandidate>& 
			      tempElectronCandidates
			      //      std::shared_ptr< reco::PFCandidateCollection > &pfElectronCandidates_  
			      ){
    isvalid_=false;
    // RunPFPhoton has to set isvalid_ to TRUE in case it finds a valid candidate
    // ... TODO: maybe can be replaced by checking for the size of the CandCollection.....
    permElectronCandidates_.clear();
    match_ind.clear();
    RunPFPhoton(blockRef,
		active,
		pfPhotonCandidates,
		pfPhotonExtraCandidates,
		tempElectronCandidates
		);
    int ind=0;
    int matches=match_ind.size();
    
    for ( std::vector<reco::PFCandidate>::const_iterator ec=tempElectronCandidates.begin();   ec != tempElectronCandidates.end(); ++ec){
      bool matched=false;
      for(int i=0; i<matches; i++)
	{
	  if(ind==match_ind[i])
	    {
	      matched=true; 
	      //std::cout<<"This is matched in .h "<<*ec<<std::endl; 
		break;
	    }
	}
      ++ind;
      if(matched)continue;
      permElectronCandidates_.push_back(*ec);	  
      //std::cout<<"This is NOT matched in .h "<<*ec<<std::endl; 
    }
    
    match_ind.clear();
    
    tempElectronCandidates.clear(); 
    for ( std::vector<reco::PFCandidate>::const_iterator ec=permElectronCandidates_.begin();   ec != permElectronCandidates_.end(); ++ec)tempElectronCandidates.push_back(*ec);
    permElectronCandidates_.clear();
    
    return isvalid_;
  };
  
private: 

  enum verbosityLevel {
    Silent,
    Summary,
    Chatty
  };
  
  
  bool isvalid_;                               // is set to TRUE when a valid PhotonCandidate is found in a PFBlock
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
  
  boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_;
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
  void RunPFPhoton(const reco::PFBlockRef&  blockRef,
		   std::vector< bool >& active,

		   std::unique_ptr<reco::PFCandidateCollection> &pfPhotonCandidates,
		   std::vector<reco::PFCandidatePhotonExtra>& 
		   pfPhotonExtraCandidates,
		   //  std::unique_ptr<reco::PFCandidateCollection> 
		   //&pfElectronCandidates_
		   std::vector<reco::PFCandidate>& 
		   tempElectronCandidates
		   );

  bool EvaluateSingleLegMVA(const reco::PFBlockRef& blockref, 
			    const reco::Vertex& primaryvtx, 
			    unsigned int track_index);
  
  double ClustersPhiRMS(const std::vector<reco::CaloCluster>&PFClusters, float PFPhoPhi);
  float EvaluateLCorrMVA(reco::PFClusterRef clusterRef );
  float EvaluateGCorrMVA(const reco::PFCandidate&, const std::vector<reco::CaloCluster>& PFClusters);
  float EvaluateResMVA(const reco::PFCandidate&,const std::vector<reco::CaloCluster>& PFClusters );
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
