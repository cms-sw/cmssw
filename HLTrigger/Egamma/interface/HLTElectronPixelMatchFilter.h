#ifndef HLTElectronPixelMatchFilter_h
#define HLTElectronPixelMatchFilter_h

/** \class HLTElectronPixelMatchFilter
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class decleration
//

class HLTElectronPixelMatchFilter : public HLTFilter {

   public:
      explicit HLTElectronPixelMatchFilter(const edm::ParameterSet&);
      ~HLTElectronPixelMatchFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag candTag_;     // input tag identifying product contains filtered egammas
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;

      edm::InputTag L1IsoPixelSeedsTag_; // input tag for the pixel seed - supercluster map
      edm::EDGetTokenT<reco::ElectronSeedCollection> L1IsoPixelSeedsToken_;
      //edm::InputTag L1IsoPixelmapendcapTag_; // input tag for the pixel seed - supercluster map

      edm::InputTag L1NonIsoPixelSeedsTag_; // input tag for the pixel seed - supercluster map
      edm::EDGetTokenT<reco::ElectronSeedCollection> L1NonIsoPixelSeedsToken_;
      //edm::InputTag L1NonIsoPixelmapendcapTag_; // input tag for the pixel seed - supercluster map

      double npixelmatchcut_;     // number of pixelmatch hits
      int    ncandcut_;           // number of electrons required
      
      bool doIsolated_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTElectronPixelMatchFilter_h


