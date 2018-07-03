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
      ~HLTDeDxFilter() override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;

   private:
      bool saveTags_;              // whether to save this tag
      double minDEDx_;
      double minPT_;
      double minNOM_;
      double maxETA_;
      double minNumValidHits_;
      double maxNHitMissIn_;
      double maxNHitMissMid_;
      double maxRelTrkIsoDeltaRp3_;
      double relTrkIsoDeltaRSize_;
      double maxAssocCaloE_;
      double maxAssocCaloEDeltaRSize_;
      edm::EDGetToken inputTracksToken_;
      edm::EDGetToken caloTowersToken_;
      edm::EDGetToken inputdedxToken_;
      edm::InputTag caloTowersTag_;
      edm::InputTag inputTracksTag_;
      edm::InputTag inputdedxTag_;
      edm::InputTag thisModuleTag_;
};

#endif //HLTDeDxFilter_h



