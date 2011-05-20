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
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
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
	       const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration); 

  //destructor
  ~PFPhotonAlgo(){delete tmvaReader_;};
  
  //check candidate validity
  bool isPhotonValidCandidate(const reco::PFBlockRef&  blockRef,
			      std::vector< bool >&  active,
			      std::auto_ptr< reco::PFCandidateCollection > &pfPhotonCandidates,
			      std::vector<reco::PFCandidatePhotonExtra>& pfPhotonExtraCandidates,
			      std::auto_ptr< reco::PFCandidateCollection > & // this is dummy for noe, can later be used to pass electrons if needed
			      )
  {
    isvalid_=false;

    // RunPFPhoton has to set isvalid_ to TRUE in case it finds a valid candidate
    // ... TODO: maybe can be replaced by checking for the size of the CandCollection.....
    RunPFPhoton(blockRef,
		active,
		pfPhotonCandidates,
		pfPhotonExtraCandidates);

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

  float nlost, nlayers;
  float chi2, STIP, del_phi,HoverPt, EoverPt, track_pt;
  double mvaValue;

  void RunPFPhoton(const reco::PFBlockRef&  blockRef,
		   std::vector< bool >& active,
		   std::auto_ptr< reco::PFCandidateCollection > &pfPhotonCandidates,
		   std::vector<reco::PFCandidatePhotonExtra>& pfPhotonExtraCandidates);

  bool EvaluateSingleLegMVA(const reco::PFBlockRef& blockref, 
			    const reco::Vertex& primaryvtx, 
			    unsigned int track_index);
  
};



#endif
