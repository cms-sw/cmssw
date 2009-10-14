#ifndef PFProducer_PFConversionAlgo_H
#define PFProducer_PFConversionAlgo_H

#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "TMVA/Reader.h"
#include <iostream>

namespace reco { 
  class PFCandidate;
}

class PFConversionAlgo {
 public:
  
  //constructor
  //  PFConversionAlgo(const reco::PFBlockRef&  blockRef, std::vector<bool>&  active);
  PFConversionAlgo();
		
  
  //destructor
  ~PFConversionAlgo(){;};
  
  //check candidate validity
  bool isConversionValidCandidate(const reco::PFBlockRef&  blockRef,
				std::vector<bool>&  active)
  {
    isvalid_=false;
    runPFConversion(blockRef,active);
    return isvalid_;
  };
  
  //get electron PFCandidate
  std::vector<reco::PFCandidate> conversionCandidates() {return conversionCandidate_;};
  
  
 private: 
  //  typedef  std::vector<std::pair< unsigned int, std::vector<unsigned int> > > AssMap;

  typedef  std::multimap<unsigned, std::vector<unsigned> > AssMap;

  void runPFConversion(const reco::PFBlockRef&  blockRef, std::vector<bool>& active);

  //void SetIDOutputs(const reco::PFBlock& block);

  bool setLinks(const reco::PFBlockRef& blockRef, AssMap& assToConv,  std::vector<bool>& active );
  void setCandidates(const reco::PFBlockRef& blockref, AssMap& assToConv);
  void setActive(const reco::PFBlockRef& blockRef, AssMap& assToConv,std::vector<bool>& active ) ;

  
  std::vector<reco::PFCandidate> conversionCandidate_;
  bool isvalid_;
};
#endif
