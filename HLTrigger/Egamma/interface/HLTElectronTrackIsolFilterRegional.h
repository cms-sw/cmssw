#ifndef HLTElectronTrackIsolFilterRegional_h
#define HLTElectronTrackIsolFilterRegional_h

/** \class HLTElectronTrackIsolFilterRegional
 * $Id: HLTElectronTrackIsolFilterRegional.h,v 1.1 2007/01/26 10:38:13 monicava Exp $
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
      double pttrackisolcut_;   // pt of Tracks in cone around candidate
      int    ncandcut_;        // number of electrons required
};

#endif //HLTElectronTrackIsolFilterRegional_h


