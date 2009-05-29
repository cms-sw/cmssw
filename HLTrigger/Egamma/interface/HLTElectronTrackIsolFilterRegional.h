#ifndef HLTElectronTrackIsolFilterRegional_h
#define HLTElectronTrackIsolFilterRegional_h

/** \class HLTElectronTrackIsolFilterRegional
 * $Id: HLTElectronTrackIsolFilterRegional.h,v 1.4 2008/09/17 15:40:10 ghezzi Exp $
 *   
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTElectronTrackIsolFilterRegional : public HLTFilter {

   public:
      explicit HLTElectronTrackIsolFilterRegional(const edm::ParameterSet&);
      ~HLTElectronTrackIsolFilterRegional();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains filtered electrons
      edm::InputTag isoTag_; // input tag identifying product contains track isolation map
      edm::InputTag nonIsoTag_; // input tag identifying product contains track isolation map
      double pttrackisolcut_;   // pt of Tracks in cone around candidate
      double pttrackisolOverEcut_;   // (pt of Tracks in cone around candidate)/pt_ele
      double pttrackisolOverE2cut_;   // (pt of Tracks in cone around candidate)/pt_ele^2
      int    ncandcut_;        // number of electrons required
      bool doIsolated_;

      bool   store_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTElectronTrackIsolFilterRegional_h


