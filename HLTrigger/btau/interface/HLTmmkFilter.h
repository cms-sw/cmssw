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
#include "DataFormats/TrackReco/interface/Track.h"

// ----------------------------------------------------------------------

namespace reco {
  class Candidate; 
}
	
class HLTmmkFilter : public HLTFilter {
 public:
  explicit HLTmmkFilter(const edm::ParameterSet&);
  ~HLTmmkFilter();
  
 private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  virtual int overlap(const reco::Candidate&, const reco::Candidate&);
  virtual double deltaPhi(double phi1, double phi2);


  edm::InputTag fJpsiLabel;
  edm::InputTag fTrackLabel; 


};
#endif
