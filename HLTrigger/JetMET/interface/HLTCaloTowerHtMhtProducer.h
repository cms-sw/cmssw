#ifndef HLTCaloTowerHtMhtProducer_h_
#define HLTCaloTowerHtMhtProducer_h_

/** \class HLTCaloTowerHtMhtProducer
 *
 *  \brief  This produces a reco::MET object that stores HT and MHT
 *  \author Steven Lowette
 *  \author Michele de Gruttola, Jia Fu Low (Nov 2013)
 *  \author Thiago Tomei (added the "HT from CaloTower code")
 *
 *  HT & MHT are calculated using input CaloTower collection.
 *  HT is stored as `sumet_`, MHT is stored as `p4_` in the output.
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h" 
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"

namespace edm {
    class ConfigurationDescriptions;
}

// Class declaration
class HLTCaloTowerHtMhtProducer : public edm::stream::EDProducer<> {
  public:
    explicit HLTCaloTowerHtMhtProducer(const edm::ParameterSet & iConfig);
    ~HLTCaloTowerHtMhtProducer();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:
    /// Use pt; otherwise, use et.
    bool usePt_;

    /// Minimum pt requirement for jets
    double minPtTowerHt_;
    double minPtTowerMht_;

    /// Maximum (abs) eta requirement for jets
    double maxEtaTowerHt_;
    double maxEtaTowerMht_;

    /// Input CaloTower collection
    edm::InputTag towersLabel_;
    edm::EDGetTokenT<CaloTowerCollection> m_theTowersToken;
};

#endif  // HLTCaloTowerHtMhtProducer_h_

