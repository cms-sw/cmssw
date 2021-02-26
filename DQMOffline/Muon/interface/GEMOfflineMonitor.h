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
  void bookDigiOccupancy(DQMStore::IBooker &, const edm::ESHandle<GEMGeometry> &);
  void bookHitOccupancy(DQMStore::IBooker &, const edm::ESHandle<GEMGeometry> &);

  void doDigiOccupancy(const edm::ESHandle<GEMGeometry> &, const edm::Handle<GEMDigiCollection> &);
  void doHitOccupancy(const edm::ESHandle<GEMGeometry> &, const edm::Handle<GEMRecHitCollection> &);

  edm::EDGetTokenT<GEMDigiCollection> digi_token_;
  edm::EDGetTokenT<GEMRecHitCollection> rechit_token_;

  bool do_digi_occupancy_;
  bool do_hit_occupancy_;

  MEMap me_digi_det_;  // TH2F, region-station
  MEMap me_hit_det_;   // TH2F, region-station
};

#endif  // DQMOffline_Muon_GEMOfflineMonitor_h
