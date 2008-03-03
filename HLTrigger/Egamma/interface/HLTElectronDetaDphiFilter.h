#ifndef HLTElectronDetaDphiFilter_h
#define HLTElectronDetaDphiFilter_h

/** \class HLTElectronDetaDphiFilter
 * $Id: HLTElectronDetaDphiFilter.h,v 1.2 2007/04/02 17:14:13 mpieri Exp $
 *   
 *
 *  \author Alessio Ghezzi (CERN & Milano-Bicocca)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTElectronDetaDphiFilter : public HLTFilter {

   public:
      explicit HLTElectronDetaDphiFilter(const edm::ParameterSet&);
      ~HLTElectronDetaDphiFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag identifying product contains filtered electrons

      double DeltaEtacut_;   // Delta eta SC-track
      double DeltaPhicut_;   // Delta phi SC-track

      int    ncandcut_;        // number of electrons required

      //      bool doIsolated_;
};

#endif //HLTElectronDetaDphiFilter_h


