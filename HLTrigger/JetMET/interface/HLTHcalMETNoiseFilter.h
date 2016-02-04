#ifndef HLTHcalMETNoiseFilter_h
#define HLTHcalMETNoiseFilter_h

/** \class HLTHcalNoiseFilter
 *
 *  \author Leonard Apanasevich (UIC), John Paul Chou (Brown)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTHcalMETNoiseFilter : public HLTFilter {
  
 public:
  explicit HLTHcalMETNoiseFilter(const edm::ParameterSet&);
  ~HLTHcalMETNoiseFilter();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
 private:
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
  double minRecHitE_, minLowHitE_, minHighHitE_;
  
  // helper function to compare noise data energies
  struct noisedatacomp {
    inline bool operator() ( const CommonHcalNoiseRBXData& t1, const CommonHcalNoiseRBXData& t2) {
      return t1.energy()>t2.energy();
    }
  };
  typedef std::set<CommonHcalNoiseRBXData, noisedatacomp> noisedataset_t;
};

#endif //HLTHcalMETNoiseFilter_h
