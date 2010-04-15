#ifndef HLTHcalMETNoiseFilter_h
#define HLTHcalMETNoiseFilter_h

/** \class HLTHcalNoiseFilter
 *
 *  \author Leonard Apanasevich (UIC), John Paul Chou (Brown)
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"

class HLTHcalMETNoiseFilter : public HLTFilter {
  
 public:
  explicit HLTHcalMETNoiseFilter(const edm::ParameterSet&);
  ~HLTHcalMETNoiseFilter();
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
 private:
  HcalNoiseAlgo noisealgo_;
  
  // parameters
  edm::InputTag HcalNoiseRBXCollectionTag_;
  int severity_;
  int maxNumRBXs_;
  int numRBXsToConsider_;
  bool needHighLevelCoincidence_;
  bool useLooseRatioFilter_;
  bool useLooseHitsFilter_;
  bool useLooseZerosFilter_;
  bool useLooseTimingFilter_;
  bool useTightRatioFilter_;
  bool useTightHitsFilter_;
  bool useTightZerosFilter_;
  bool useTightTimingFilter_;
  
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
