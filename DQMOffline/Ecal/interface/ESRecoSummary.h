#ifndef ESRecoSummary_h
#define ESRecoSummary_h

// system include files
#include <memory>

// DQM includes
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"

class ESRecoSummary : public DQMEDAnalyzer {
 public:
  explicit ESRecoSummary(const edm::ParameterSet&);
  ~ESRecoSummary() {}
  
 private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string prefixME_;

  // PRESHOWER ----------------------------------------------
  MonitorElement* h_recHits_ES_energyMax;
  MonitorElement* h_recHits_ES_time;
      
  MonitorElement* h_esClusters_energy_plane1;
  MonitorElement* h_esClusters_energy_plane2;
  MonitorElement* h_esClusters_energy_ratio;
         
 protected:
  

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::SuperClusterCollection> superClusterCollection_EE_;
  edm::EDGetTokenT<ESRecHitCollection> esRecHitCollection_;
  edm::EDGetTokenT<reco::PreshowerClusterCollection> esClusterCollectionX_ ;
  edm::EDGetTokenT<reco::PreshowerClusterCollection> esClusterCollectionY_ ;
};

#endif
