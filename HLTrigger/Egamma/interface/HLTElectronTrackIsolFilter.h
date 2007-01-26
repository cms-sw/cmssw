#ifndef HLTElectronTrackIsolFilter_h
#define HLTElectronTrackIsolFilter_h

/** \class HLTElectronTrackIsolFilter
 * $Id: HLTElectronTrackIsolFilter.h,v 1.1 2006/10/26 20:47:42 monicava Exp $
 *   
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTElectronTrackIsolFilter : public HLTFilter {

   public:
      explicit HLTElectronTrackIsolFilter(const edm::ParameterSet&);
      ~HLTElectronTrackIsolFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains filtered electrons
      edm::InputTag isoTag_; // input tag identifying product contains track isolation map
      double pttrackisolcut_;   // pt of Tracks in cone around candidate
      int    ncandcut_;        // number of electrons required
};

#endif //HLTElectronTrackIsolFilter_h


