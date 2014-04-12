#ifndef HLTHcalNoiseFilter_h
#define HLTHcalNoiseFilter_h

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

namespace edm {
  class ConfigurationDescriptions;
}

class HLTHcalNoiseFilter : public HLTFilter {
   public:
      explicit HLTHcalNoiseFilter(const edm::ParameterSet&);
      ~HLTHcalNoiseFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::EDGetTokenT<reco::CaloJetCollection> JetSourceToken_;
      edm::EDGetTokenT<reco::CaloMETCollection> MetSourceToken_;
      edm::EDGetTokenT<CaloTowerCollection> TowerSourceToken_;
      edm::InputTag JetSource_;
      edm::InputTag MetSource_;
      edm::InputTag TowerSource_;
      bool useMet_;
      bool useJet_;
      double MetCut_;
      double JetMinE_;
      double JetHCALminEnergyFraction_;
};

#endif
