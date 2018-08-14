#ifndef HLTHcalMETNoiseFilter_h
#define HLTHcalMETNoiseFilter_h

/** \class HLTHcalNoiseFilter
 *
 *  \author Leonard Apanasevich (UIC), John Paul Chou (Brown)
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"

#include "DataFormats/METReco/interface/HcalNoiseRBX.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTHcalMETNoiseFilter : public edm::EDFilter {
  
 public:
  explicit HLTHcalMETNoiseFilter(const edm::ParameterSet&);
  ~HLTHcalMETNoiseFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  bool filter(edm::Event&, const edm::EventSetup&) override;
  
 private:
  edm::EDGetTokenT<reco::HcalNoiseRBXCollection> m_theHcalNoiseToken;
  // parameters
  edm::InputTag HcalNoiseRBXCollectionTag_;
  int severity_;
  int maxNumRBXs_;
  int numRBXsToConsider_;
  bool needEMFCoincidence_;
  double minRBXEnergy_;
  double minRatio_;
  double maxRatio_;
  int minHPDHits_;
  int minRBXHits_;
  int minHPDNoOtherHits_;
  int minZeros_;
  double minHighEHitTime_;
  double maxHighEHitTime_;
  double maxRBXEMF_;

  // imported from the RecoMET/METProducers/python/hcalnoiseinfoproducer_cfi
  double minRecHitE_, minLowHitE_, minHighHitE_, minR45HitE_;

  double TS4TS5EnergyThreshold_;
  std::vector<std::pair<double, double> > TS4TS5UpperCut_;
  std::vector<std::pair<double, double> > TS4TS5LowerCut_;  
    
  // helper function to compare noise data energies
  struct noisedatacomp {
    inline bool operator() ( const CommonHcalNoiseRBXData& t1, const CommonHcalNoiseRBXData& t2) const{
      return t1.energy()>t2.energy();
    }
  };
  typedef std::set<CommonHcalNoiseRBXData, noisedatacomp> noisedataset_t;
};

#endif //HLTHcalMETNoiseFilter_h
