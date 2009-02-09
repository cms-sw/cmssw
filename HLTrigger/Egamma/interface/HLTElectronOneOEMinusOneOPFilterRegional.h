#ifndef HLTElectronOneOEMinusOneOPFilterRegional_h
#define HLTElectronOneOEMinusOneOPFilterRegional_h

/** \class HLTElectronOneOEMinusOneOPFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 * $Id: HLTElectronOneOEMinusOneOPFilterRegional.h,v 1.1 2007/08/28 15:21:51 ghezzi Exp $
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTElectronOneOEMinusOneOPFilterRegional : public HLTFilter {

   public:
      explicit HLTElectronOneOEMinusOneOPFilterRegional(const edm::ParameterSet&);
      ~HLTElectronOneOEMinusOneOPFilterRegional();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag for the RecoCandidates from the previous filter
      edm::InputTag electronIsolatedProducer_;// input tag for the producer of electrons
      edm::InputTag electronNonIsolatedProducer_;// input tag for the producer of electrons
      bool doIsolated_;
      double barrelcut_; //  Eoverp barrel
      double endcapcut_; //  Eoverp endcap
      int    ncandcut_;        // number of electrons required

      bool   store_;
      // edm::InputTag L1IsoCollTag_; 
      // edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTElectronOneOEMinusOneOPFilterRegional_h
