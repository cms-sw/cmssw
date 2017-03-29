#ifndef HLTElectronPixelMatchFilter_h
#define HLTElectronPixelMatchFilter_h

/** \class HLTElectronPixelMatchFilter
 *
 *  \author Aidan Randle-Condle (ULB)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
namespace edm {
  class ConfigurationDescriptions;
}

class HLTElectronPixelMatchFilter : public HLTFilter {

 public:
  explicit HLTElectronPixelMatchFilter(const edm::ParameterSet&);
  ~HLTElectronPixelMatchFilter();
  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
 private:
  float calDPhi1Sq(reco::ElectronSeedCollection::const_iterator seed, int charge)const;
  float calDPhi2Sq(reco::ElectronSeedCollection::const_iterator seed, int charge)const;
  float calDZ2Sq(reco::ElectronSeedCollection::const_iterator seed, int charge)const;
  int getNrOfMatches(edm::Handle<reco::ElectronSeedCollection>& eleSeeds,
		     reco::SuperClusterRef& candSCRef)const;

  
  edm::InputTag candTag_;     // input tag identifying product contains filtered egammas
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> candToken_;
  
  edm::InputTag l1PixelSeedsTag_; // input tag for the pixel seed - supercluster map
  edm::EDGetTokenT<reco::ElectronSeedCollection> l1PixelSeedsToken_;
  
  double npixelmatchcut_;     // number of pixelmatch hits
  int    ncandcut_;           // number of electrons required
  
  edm::InputTag l1EGTag_;
  
  // cuts on s2
  float s2BarrelThres_ ;
  float s2InterThres_;
  float s2ForwardThres_;

  //parameters to create s (divide by param)
  float sPhi1B_ ;
  float sPhi1I_ ;
  float sPhi1F_ ;
  float sPhi2B_ ;
  float sPhi2I_ ;
  float sPhi2F_ ;
  float sZ2B_ ;
  float sR2I_ ;
  float sR2F_ ;
  
  bool isPixelVeto_;
  bool useS_ ;
};

#endif //HLTElectronPixelMatchFilter_h


