#ifndef DQMOffline_Muon_GEMOfflineMonitor_h
#define DQMOffline_Muon_GEMOfflineMonitor_h

#include "DQMOffline/Muon/interface/GEMOfflineDQMBase.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"

class GEMOfflineMonitor : public GEMOfflineDQMBase {
public:
  explicit GEMOfflineMonitor(const edm::ParameterSet &);
  ~GEMOfflineMonitor() override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  void bookDetectorOccupancy(
      DQMStore::IBooker &, const GEMStation *, const MEMapKey1 &, const TString &, const TString &);

  edm::EDGetTokenT<GEMDigiCollection> digi_token_;
  edm::EDGetTokenT<GEMRecHitCollection> rechit_token_;

  std::string log_category_;

  MEMap1 me_digi_det_;
  MEMap1 me_hit_det_;
};

#endif  // DQMOffline_Muon_GEMOfflineMonitor_h
