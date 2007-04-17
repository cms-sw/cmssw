#ifndef HLTMuonIsoFilter_h
#define HLTMuonIsoFilter_h

/** \class HLTMuonIsoFilter
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing
 *  the isolation filtering for HLT muons
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

class HLTMuonIsoFilter : public HLTFilter {

   public:
      explicit HLTMuonIsoFilter(const edm::ParameterSet&);
      ~HLTMuonIsoFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying muon container
      edm::InputTag isoTag_;  // input tag identifying isolation map
      
      int    min_N_;          // minimum number of muons to fire the trigger
};

#endif //HLTMuonIsoFilter_h
