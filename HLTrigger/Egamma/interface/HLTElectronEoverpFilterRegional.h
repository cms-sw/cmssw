#ifndef HLTElectronEoverpFilterRegional_h
#define HLTElectronEoverpFilterRegional_h

/** \class HLTElectronEoverpFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 * $Id: HLTElectronEoverpFilterRegional.h,v 1.1 2007/01/26 10:38:13 monicava Exp $
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTElectronEoverpFilterRegional : public HLTFilter {

   public:
      explicit HLTElectronEoverpFilterRegional(const edm::ParameterSet&);
      ~HLTElectronEoverpFilterRegional();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_; // input tag for the RecoCandidates from the previous filter
      edm::InputTag electronProducer_;// input tag for the producer of electrons
      double eoverpbarrelcut_; //  Eoverp barrel
      double eoverpendcapcut_; //  Eoverp endcap
      int    ncandcut_;        // number of electrons required
};

#endif //HLTElectronEoverpFilterRegional_h
