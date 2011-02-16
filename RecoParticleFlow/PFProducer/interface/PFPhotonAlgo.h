#ifndef PFProducer_PFPhotonAlgo_H
#define PFProducer_PFPhotonAlgo_H

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "TMVA/Reader.h"
#include <iostream>


class PFSCEnergyCalibration;

namespace reco {
  class PFCandidate;
  class PFCandidateCollectioon;
}

class PFPhotonAlgo {
 public:
  
  //constructor
  PFPhotonAlgo(); 

  //destructor
  ~PFPhotonAlgo(){};
  
  //check candidate validity
  bool isPhotonValidCandidate(const reco::PFBlockRef&  blockRef,
			      std::vector< bool >&  active,
			      std::auto_ptr< reco::PFCandidateCollection > &pfPhotonCandidates,
			      std::auto_ptr< reco::PFCandidateCollection > & // this is dummy for noe, can later be used to pass electrons if needed
			      )
  {
    isvalid_=false;

    // RunPFPhoton has to set isvalid_ to TRUE in case it finds a valid candidate
    // ... TODO: maybe can be replaced by checking for the size of the CandCollection.....
    RunPFPhoton(blockRef,
		active,
		pfPhotonCandidates);

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
  
  void RunPFPhoton(const reco::PFBlockRef&  blockRef,
		   std::vector< bool >& active,
		   std::auto_ptr< reco::PFCandidateCollection > &pfPhotonCandidates);
  
};


#endif
