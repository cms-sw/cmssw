#ifndef PFProducer_PFPhotonAlgo_H
#define PFProducer_PFPhotonAlgo_H

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtraFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
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
  //check candidate validity
  bool isPhotonValidCandidate(const reco::PFBlockRef&  blockRef,
			      std::vector< bool >&  active,
			      std::auto_ptr< reco::PFCandidateCollection > &pfPhotonCandidates,
			      std::vector<reco::PFCandidatePhotonExtra>& pfPhotonExtraCandidates,
			      std::vector<reco::PFCandidate>& 
			      tempElectronCandidates
			      //      std::auto_ptr< reco::PFCandidateCollection > &pfElectronCandidates_  
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
    bool matched=false;
    int matches=match_ind.size();
    
    for ( std::vector<reco::PFCandidate>::const_iterator ec=tempElectronCandidates.begin();   ec != tempElectronCandidates.end(); ++ec){
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
  reco::Vertex       primaryVertex_;
  TMVA::Reader *tmvaReader_;
  const GBRForest *ReaderLC_;
  const GBRForest *ReaderGC_;
  const GBRForest *ReaderRes_;
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
  float CrysPhi_, CrysEta_, CrysIPhi_, CrysIEta_, VtxZ_, ClusPhi_, ClusEta_, 
    ClusR9_, Clus5x5ratio_, PFCrysPhiCrack_, PFCrysEtaCrack_, logPFClusE_, e3x3_;
  float EB;
  //Cluster Shapes:
  float eSeed_, e1x3_,e3x1_, e1x5_, e2x5Top_,  e2x5Bottom_, e2x5Left_,  e2x5Right_ ; 
  float e2x5Max_;
  //For Global Corrections:
  float PFPhoEta_, PFPhoPhi_, PFPhoR9_, SCPhiWidth_, SCEtaWidth_, PFPhoEt_, RConv_;
  float dEta_, dPhi_, LowClusE_, nPFClus_;
  
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

		   std::auto_ptr< reco::PFCandidateCollection > &pfPhotonCandidates,
		   std::vector<reco::PFCandidatePhotonExtra>& 
		   pfPhotonExtraCandidates,
		   //  std::auto_ptr< reco::PFCandidateCollection > 
		   //&pfElectronCandidates_
		   std::vector<reco::PFCandidate>& 
		   tempElectronCandidates
		   );

  bool EvaluateSingleLegMVA(const reco::PFBlockRef& blockref, 
			    const reco::Vertex& primaryvtx, 
			    unsigned int track_index);
  

  void GetCrysCoordinates(reco::PFClusterRef clusterRef);
  void fill5x5Map(reco::PFClusterRef clusterRef);
  float EvaluateLCorrMVA(reco::PFClusterRef clusterRef );
  float EvaluateGCorrMVA(reco::PFCandidate);
  float EvaluateResMVA(reco::PFCandidate);
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
