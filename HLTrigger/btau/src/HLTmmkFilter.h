#ifndef HLTmmkFilter_h
#define HLTmmkFilter_h
//
// Package:    HLTstaging
// Class:      HLTmmkFilter
// 
/**\class HLTmmkFilter 

 HLT Filter for b to (mumu) + X

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Nicolo Magini
//         Created:  Thu Nov  9 17:55:31 CET 2006
// Modified by Lotte Wilke
// Last Modification: 13.02.2007
//


// system include files
#include <memory>

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

// ----------------------------------------------------------------------

namespace reco {
  class Candidate; 
}
	
class HLTmmkFilter : public HLTFilter {
 public:
  explicit HLTmmkFilter(const edm::ParameterSet&);
  ~HLTmmkFilter();
  
 private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual int overlap(const reco::Candidate&, const reco::Candidate&);
  
  edm::InputTag muCandLabel_;
  edm::InputTag trkCandLabel_; 
  
  const double thirdTrackMass_;
  const double maxEta_;
  const double minPt_;
  const double minInvMass_;
  const double maxInvMass_;
  const double maxNormalisedChi2_;
  const double minLxySignificance_;
  const double minCosinePointingAngle_;
  const bool fastAccept_;
	bool saveTags_;
	edm::InputTag beamSpotTag_;

};
#endif
