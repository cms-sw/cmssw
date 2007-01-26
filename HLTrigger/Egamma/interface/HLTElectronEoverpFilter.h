#ifndef HLTElectronEoverpFilter_h
#define HLTElectronEoverpFilter_h

/** \class HLTElectronEoverpFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 * $Id: HLTElectronEoverpFilter.h,v 1.1 2006/11/09 22:48:43 monicava Exp $
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTElectronEoverpFilter : public HLTFilter {

   public:
      explicit HLTElectronEoverpFilter(const edm::ParameterSet&);
      ~HLTElectronEoverpFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains electrons
      double eoverpbarrelcut_; //  Eoverp barrel
      double eoverpendcapcut_; //  Eoverp endcap
      int    ncandcut_;        // number of electrons required
};

#endif //HLTElectronEoverpFilter_h
