// system include files
#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>

// user include files
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigi.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigi.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCluster.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

class CSCDigiFilter : public edm::stream::EDProducer<> {
public:
  explicit CSCDigiFilter(const edm::ParameterSet &);
  ~CSCDigiFilter() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  template <typename T, typename C = MuonDigiCollection<CSCDetId, T>>
  void filterDigis(edm::Event &event, edm::EDGetTokenT<C> &digiToken, std::unique_ptr<C> &filteredDigis);

  // the collections
  edm::EDGetTokenT<CSCStripDigiCollection> stripDigiToken_;
  edm::EDGetTokenT<CSCWireDigiCollection> wireDigiToken_;
  edm::EDGetTokenT<CSCComparatorDigiCollection> compDigiToken_;
  edm::EDGetTokenT<CSCCLCTDigiCollection> clctDigiToken_;
  edm::EDGetTokenT<CSCALCTDigiCollection> alctDigiToken_;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> lctDigiToken_;
  edm::EDGetTokenT<CSCShowerDigiCollection> showerDigiToken_;
  edm::EDGetTokenT<GEMPadDigiClusterCollection> gemPadDigiClusterToken_;

  /* which chambers to select or mask? format example "ME+2/1/15"
     behavior:
     if maskedChambers_ is empty, no chambers are masked
     if selectedChambers_ is empty, all chambers are selected
  */
  std::vector<std::string> maskedChambers_;
  std::vector<std::string> selectedChambers_;

  // options to filter Run-3 and Phase-2 collections
  bool useGEMs_;
  bool useShowers_;
};

CSCDigiFilter::CSCDigiFilter(const edm::ParameterSet &iConfig)
    : stripDigiToken_(consumes<CSCStripDigiCollection>(iConfig.getParameter<edm::InputTag>("stripDigiTag"))),
      wireDigiToken_(consumes<CSCWireDigiCollection>(iConfig.getParameter<edm::InputTag>("wireDigiTag"))),
      compDigiToken_(consumes<CSCComparatorDigiCollection>(iConfig.getParameter<edm::InputTag>("compDigiTag"))),
      clctDigiToken_(consumes<CSCCLCTDigiCollection>(iConfig.getParameter<edm::InputTag>("clctDigiTag"))),
      alctDigiToken_(consumes<CSCALCTDigiCollection>(iConfig.getParameter<edm::InputTag>("alctDigiTag"))),
      lctDigiToken_(consumes<CSCCorrelatedLCTDigiCollection>(iConfig.getParameter<edm::InputTag>("lctDigiTag"))),
      maskedChambers_(iConfig.getParameter<std::vector<std::string>>("maskedChambers")),
      selectedChambers_(iConfig.getParameter<std::vector<std::string>>("selectedChambers")),
      useGEMs_(iConfig.getParameter<bool>("useGEMs")),
      useShowers_(iConfig.getParameter<bool>("useShowers")) {
  if (useGEMs_)
    gemPadDigiClusterToken_ =
        consumes<GEMPadDigiClusterCollection>(iConfig.getParameter<edm::InputTag>("gemPadClusterDigiTag"));
  if (useShowers_)
    showerDigiToken_ = consumes<CSCShowerDigiCollection>(iConfig.getParameter<edm::InputTag>("showerDigiTag"));

  produces<CSCStripDigiCollection>("MuonCSCStripDigi");
  produces<CSCWireDigiCollection>("MuonCSCWireDigi");
  produces<CSCComparatorDigiCollection>("MuonCSCComparatorDigi");
  produces<CSCCLCTDigiCollection>("MuonCSCCLCTDigi");
  produces<CSCALCTDigiCollection>("MuonCSCALCTDigi");
  produces<CSCCorrelatedLCTDigiCollection>("MuonCSCCorrelatedLCTDigi");
  if (useGEMs_)
    produces<GEMPadDigiClusterCollection>("MuonGEMPadDigiCluster");
  if (useShowers_)
    produces<CSCShowerDigiCollection>("MuonCSCShowerDigi");
}

void CSCDigiFilter::produce(edm::Event &event, const edm::EventSetup &conditions) {
  std::unique_ptr<CSCStripDigiCollection> filteredStripDigis(new CSCStripDigiCollection());
  std::unique_ptr<CSCWireDigiCollection> filteredWireDigis(new CSCWireDigiCollection());
  std::unique_ptr<CSCComparatorDigiCollection> filteredCompDigis(new CSCComparatorDigiCollection());
  std::unique_ptr<CSCCLCTDigiCollection> filteredCLCTDigis(new CSCCLCTDigiCollection());
  std::unique_ptr<CSCALCTDigiCollection> filteredALCTDigis(new CSCALCTDigiCollection());
  std::unique_ptr<CSCCorrelatedLCTDigiCollection> filteredLCTDigis(new CSCCorrelatedLCTDigiCollection());
  std::unique_ptr<CSCShowerDigiCollection> filteredShowerDigis(new CSCShowerDigiCollection());
  std::unique_ptr<GEMPadDigiClusterCollection> filteredClusterDigis(new GEMPadDigiClusterCollection());

  // filter the collections
  filterDigis<CSCStripDigi>(event, stripDigiToken_, filteredStripDigis);
  filterDigis<CSCWireDigi>(event, wireDigiToken_, filteredWireDigis);
  filterDigis<CSCComparatorDigi>(event, compDigiToken_, filteredCompDigis);
  filterDigis<CSCCLCTDigi>(event, clctDigiToken_, filteredCLCTDigis);
  filterDigis<CSCALCTDigi>(event, alctDigiToken_, filteredALCTDigis);
  filterDigis<CSCCorrelatedLCTDigi>(event, lctDigiToken_, filteredLCTDigis);
  if (useGEMs_)
    filterDigis<GEMPadDigiCluster>(event, gemPadDigiClusterToken_, filteredClusterDigis);
  if (useShowers_)
    filterDigis<CSCShowerDigi>(event, showerDigiToken_, filteredShowerDigis);

  // put the new collections into the event
  event.put(std::move(filteredStripDigis), "MuonCSCStripDigi");
  event.put(std::move(filteredWireDigis), "MuonCSCWireDigi");
  event.put(std::move(filteredCompDigis), "MuonCSCComparatorDigi");
  event.put(std::move(filteredCLCTDigis), "MuonCSCCLCTDigi");
  event.put(std::move(filteredALCTDigis), "MuonCSCALCTDigi");
  event.put(std::move(filteredLCTDigis), "MuonCSCCorrelatedLCTDigi");
  if (useShowers_)
    event.put(std::move(filteredShowerDigis), "MuonCSCShowerDigi");
  if (useGEMs_)
    event.put(std::move(filteredClusterDigis), "MuonGEMPadDigiCluster");
}

template <typename T, typename C>
void CSCDigiFilter::filterDigis(edm::Event &event, edm::EDGetTokenT<C> &digiToken, std::unique_ptr<C> &filteredDigis) {
  if (!digiToken.isUninitialized()) {
    auto const &digis = event.get(digiToken);

    for (const auto &j : digis) {
      // use base class here
      const DetId &detId = j.first;

      CSCDetId chId;

      if (std::is_same<C, GEMPadDigiClusterCollection>::value) {
        // explicit downcast
        const GEMDetId gemDetId(detId);

        const int zendcap(gemDetId.region() == 1 ? 1 : 2);
        // chamber id
        chId = CSCDetId(zendcap, gemDetId.station(), 1, gemDetId.chamber());

      } else {
        // chamber id
        chId = CSCDetId(detId).chamberId();
      }

      // check for masked chambers
      if (std::find(maskedChambers_.begin(), maskedChambers_.end(), chId.chamberName()) != maskedChambers_.end()) {
        continue;
      }
      // check for selected chambers
      // ignore this when selectedChambers_ is empty
      if (!selectedChambers_.empty()) {
        if (std::find(selectedChambers_.begin(), selectedChambers_.end(), chId.chamberName()) ==
            selectedChambers_.end()) {
          continue;
        }
      }
      // add the digis that are not ignored
      for (auto digiItr = j.second.first; digiItr != j.second.second; ++digiItr) {
        if (std::is_same<C, GEMPadDigiClusterCollection>::value) {
          filteredDigis->insertDigi(GEMDetId(detId), *digiItr);
        } else {
          filteredDigis->insertDigi(CSCDetId(detId), *digiItr);
        }
      }
    }
  }
}

void CSCDigiFilter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("stripDigiTag", {"muonCSCDigis:MuonCSCStripDigi"});
  desc.add<edm::InputTag>("wireDigiTag", {"muonCSCDigis:MuonCSCWireDigi"});
  desc.add<edm::InputTag>("compDigiTag", {"muonCSCDigis:MuonCSCComparatorDigi"});
  desc.add<edm::InputTag>("alctDigiTag", {"muonCSCDigis:MuonCSCALCTDigi"});
  desc.add<edm::InputTag>("clctDigiTag", {"muonCSCDigis:MuonCSCCLCTDigi"});
  desc.add<edm::InputTag>("lctDigiTag", {"muonCSCDigis:MuonCSCCorrelatedLCTDigi"});
  desc.add<edm::InputTag>("showerDigiTag", {"muonCSCDigis:MuonCSCShowerDigi"});
  desc.add<edm::InputTag>("gemPadClusterDigiTag", {"muonCSCDigis:MuonGEMPadDigiCluster"});
  desc.add<std::vector<std::string>>("maskedChambers");
  desc.add<std::vector<std::string>>("selectedChambers");
  desc.add<bool>("useGEMs", false);
  desc.add<bool>("useShowers", false);
  descriptions.add("cscDigiFilterDef", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(CSCDigiFilter);
