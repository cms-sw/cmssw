#ifndef HLTElectronTrackIsolFilterRegional_h
#define HLTElectronTrackIsolFilterRegional_h

/** \class HLTElectronTrackIsolFilterRegional
 * $Id: HLTElectronTrackIsolFilterRegional.h,v 1.2 2007/04/02 17:14:13 mpieri Exp $
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
      int    ncandcut_;        // number of electrons required
      bool doIsolated_;

      bool   store_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTElectronTrackIsolFilterRegional_h


