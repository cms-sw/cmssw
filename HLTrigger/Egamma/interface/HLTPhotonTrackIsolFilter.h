#ifndef HLTPhotonTrackIsolFilter_h
#define HLTPhotonTrackIsolFilter_h

/** \class HLTPhotonTrackIsolFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTPhotonTrackIsolFilter : public HLTFilter {

   public:
      explicit HLTPhotonTrackIsolFilter(const edm::ParameterSet&);
      ~HLTPhotonTrackIsolFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains filtered photons
      edm::InputTag isoTag_; // input tag identifying product contains track isolation map
      edm::InputTag nonIsoTag_; // input tag identifying product contains track isolation map
      double numtrackisolcut_;   // Number of Tracks in cone around candidate
      int    ncandcut_;        // number of photons required
      bool doIsolated_;

      bool   store_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTPhotonTrackIsolFilter_h


