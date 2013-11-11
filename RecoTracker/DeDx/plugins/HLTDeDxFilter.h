#ifndef HLTDEDxFilter_h
#define HLTDEDxFilter_h

/** \class HLTDeDxFilter
 *
 *  \author Gheorghe Lungu
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
   class ConfigurationDescriptions;
}


//
// class declaration
//

class HLTDeDxFilter : public HLTFilter {

   public:
      explicit HLTDeDxFilter(const edm::ParameterSet&);
      ~HLTDeDxFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      bool saveTags_;              // whether to save this tag
      double minDEDx_;
      double minPT_;
      double minNOM_;
      double maxETA_;
      edm::EDGetToken inputTracksToken_;
      edm::EDGetToken inputdedxToken_;
      edm::InputTag inputTracksTag_;
      edm::InputTag inputdedxTag_;
      edm::InputTag thisModuleTag_;
};

#endif //HLTDeDxFilter_h



