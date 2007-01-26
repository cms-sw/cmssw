#ifndef HLTElectronPixelMatchFilter_h
#define HLTElectronPixelMatchFilter_h

/** \class HLTElectronPixelMatchFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTElectronPixelMatchFilter : public HLTFilter {

   public:
      explicit HLTElectronPixelMatchFilter(const edm::ParameterSet&);
      ~HLTElectronPixelMatchFilter();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag candTag_;     // input tag identifying product contains filtered egammas
      edm::InputTag pixelmapbarrelTag_; // input tag for the pixel seed - supercluster map
      edm::InputTag pixelmapendcapTag_; // input tag for the pixel seed - supercluster map
      double npixelmatchcut_;     // number of pixelmatch hits
      int    ncandcut_;           // number of electrons required

};

#endif //HLTElectronPixelMatchFilter_h


