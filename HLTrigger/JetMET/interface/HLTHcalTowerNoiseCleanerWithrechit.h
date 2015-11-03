#ifndef HLTHcalTowerNoiseCleanerWithrechit_h
#define HLTHcalTowerNoiseCleanerWithrechit_h

/** \class HLTHcalNoiseCleaner
 *
 *  \author Alexander Mott (Caltech), Leonard Apanasevich (UIC), John Paul Chou (Brown)
 *
 */
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"   
#include "RecoMET/METAlgorithms/interface/HcalNoiseAlgo.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/METReco/interface/HcalNoiseRBX.h"

namespace edm {
   class ConfigurationDescriptions;
}

class HLTHcalTowerNoiseCleanerWithrechit : public edm::stream::EDProducer<> {
  
 public:
  explicit HLTHcalTowerNoiseCleanerWithrechit(const edm::ParameterSet&);
  ~HLTHcalTowerNoiseCleanerWithrechit();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  edm::EDGetTokenT<reco::HcalNoiseRBXCollection> m_theHcalNoiseToken;
  edm::EDGetTokenT<CaloTowerCollection> m_theCaloTowerCollectionToken;
  // parameters
  edm::InputTag HcalNoiseRBXCollectionTag_;
  edm::InputTag TowerCollectionTag_;
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

  // Rechit-r45 filter parameters:
  std::vector<double> hltMinRBXRechitR45Cuts_;

  // helper function to compare noise data energies
  struct noisedatacomp {
    inline bool operator() ( const CommonHcalNoiseRBXData& t1, const CommonHcalNoiseRBXData& t2) {
      return t1.energy()>t2.energy();
    }
  };
  typedef std::set<CommonHcalNoiseRBXData, noisedatacomp> noisedataset_t;
};

#endif //HLTHcalTowerNoiseCleanerWithrechit_h
