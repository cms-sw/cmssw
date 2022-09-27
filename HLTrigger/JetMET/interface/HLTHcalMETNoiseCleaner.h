#ifndef HLTHcalMETNoiseCleaner_h
#define HLTHcalMETNoiseCleaner_h

/** \class HLTHcalNoiseCleaner
 *
 *  \author Alexander Mott (Caltech), Leonard Apanasevich (UIC), John Paul Chou (Brown)
 *
 */

#include "FWCore/Framework/interface/global/EDFilter.h"
#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"
#include "DataFormats/METReco/interface/HcalNoiseRBX.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTHcalMETNoiseCleaner : public edm::global::EDFilter<> {
public:
  explicit HLTHcalMETNoiseCleaner(const edm::ParameterSet&);
  ~HLTHcalMETNoiseCleaner() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  edm::EDGetTokenT<reco::CaloMETCollection> m_theCaloMetToken;
  edm::EDGetTokenT<reco::HcalNoiseRBXCollection> m_theHcalNoiseToken;
  // parameters
  const edm::InputTag HcalNoiseRBXCollectionTag_;
  const edm::InputTag CaloMetCollectionTag_;
  const double CaloMetCut_;
  const int severity_;
  const int maxNumRBXs_;
  const int numRBXsToConsider_;
  const bool accept2NoiseRBXEvents_;
  const bool needEMFCoincidence_;
  const double minRBXEnergy_;
  const double minRatio_;
  const double maxRatio_;
  const int minHPDHits_;
  const int minRBXHits_;
  const int minHPDNoOtherHits_;
  const int minZeros_;
  const double minHighEHitTime_;
  const double maxHighEHitTime_;
  const double maxRBXEMF_;

  // imported from the RecoMET/METProducers/python/hcalnoiseinfoproducer_cfi
  const double minRecHitE_;
  const double minLowHitE_;
  const double minHighHitE_;
  const double minR45HitE_;

  const double TS4TS5EnergyThreshold_;
  std::vector<std::pair<double, double> > TS4TS5UpperCut_;
  std::vector<std::pair<double, double> > TS4TS5LowerCut_;

  reco::CaloMET BuildCaloMet(float sumet, float pt, float phi) const;

  // helper function to compare noise data energies
  struct noisedatacomp {
    inline bool operator()(const CommonHcalNoiseRBXData& t1, const CommonHcalNoiseRBXData& t2) const {
      return t1.energy() > t2.energy();
    }
  };
  typedef std::set<CommonHcalNoiseRBXData, noisedatacomp> noisedataset_t;
};

#endif  //HLTHcalMETNoiseCleaner_h
