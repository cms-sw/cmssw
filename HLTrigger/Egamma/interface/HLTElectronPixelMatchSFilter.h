#ifndef HLTElectronPixelMatchSFilter_h
#define HLTElectronPixelMatchSFilter_h

/** \class HLTElectronPixelMatchSFilter
 *
 *  \author Aidan Randle-Condle (ULB)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

//
// class decleration
//

class HLTElectronPixelMatchSFilter : public HLTFilter {

   public:
      explicit HLTElectronPixelMatchSFilter(const edm::ParameterSet&);
      ~HLTElectronPixelMatchSFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      
 private:
      edm::InputTag candTag_;     // input tag identifying product contains filtered egammas
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;

      edm::InputTag L1IsoPixelSeedsTag_; // input tag for the pixel seed - supercluster map
      //edm::InputTag L1IsoPixelmapendcapTag_; // input tag for the pixel seed - supercluster map
      edm::EDGetTokenT<reco::ElectronSeedCollection> L1IsoPixelSeedsToken_;

      edm::InputTag L1NonIsoPixelSeedsTag_; // input tag for the pixel seed - supercluster map
      //edm::InputTag L1NonIsoPixelmapendcapTag_; // input tag for the pixel seed - supercluster map
      edm::EDGetTokenT<reco::ElectronSeedCollection> L1NonIsoPixelSeedsToken_;

      double npixelmatchcut_;     // number of pixelmatch hits
      int    ncandcut_;           // number of electrons required
      
      bool doIsolated_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_;
      
      float calculate_s2(reco::ElectronSeedCollection::const_iterator, int) ;
      
      // S parameter values
      // Divide by s_a_
      float s2_threshold_ ;
      
      float s_a_phi1B_ ;
      float s_a_phi1I_ ;
      float s_a_phi1F_ ;
      float s_a_phi2B_ ;
      float s_a_phi2I_ ;
      float s_a_phi2F_ ;
      float s_a_zB_ ;
      float s_a_rI_ ;
      float s_a_rF_ ;
      
      // Multiply by s_b_ (marginally quicker)
      float s_b_phi1B_ ;
      float s_b_phi1I_ ;
      float s_b_phi1F_ ;
      float s_b_phi2B_ ;
      float s_b_phi2I_ ;
      float s_b_phi2F_ ;
      float s_b_zB_ ;
      float s_b_rI_ ;
      float s_b_rF_ ;
      
      bool use_s_ ;
};

#endif //HLTElectronPixelMatchSFilter_h


