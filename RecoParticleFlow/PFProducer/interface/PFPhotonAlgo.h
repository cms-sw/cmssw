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
#include "TMVA/Reader.h"
#include <iostream>


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
	       const reco::Vertex& primary,
	       const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
               double sumPtTrackIsoForPhoton,
               double sumPtTrackIsoSlopeForPhoton
); 

  //destructor
  ~PFPhotonAlgo(){delete tmvaReader_;};
  
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
  reco::Vertex       primaryVertex_;
  TMVA::Reader *tmvaReader_;
  boost::shared_ptr<PFEnergyCalibration> thePFEnergyCalibration_;
  double sumPtTrackIsoForPhoton_;
  double sumPtTrackIsoSlopeForPhoton_;
  std::vector<int>match_ind;
  //std::auto_ptr< reco::PFCandidateCollection > permElectronCandidates_;

  std::vector< reco::PFCandidate >permElectronCandidates_;
  float nlost, nlayers;
  float chi2, STIP, del_phi,HoverPt, EoverPt, track_pt;
  double mvaValue;
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
  

  void EarlyConversion(
		       //std::auto_ptr< reco::PFCandidateCollection > 
		       //&pfElectronCandidates_,
		       std::vector<reco::PFCandidate>& 
		       tempElectronCandidates,
		       const reco::PFBlockElementSuperCluster* sc
		       );
};

#endif
