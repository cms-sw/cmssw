/** \class L1TkMuStubFlatTableProducer
 *
 *  \brief Flat table of the muon-chamber stubs of the L1 tracker muons
 *
 *  Each row is one stub. The columns are the eta, phi and depth regions, the type and the
 *  quality of the stub, and the index of its L1 tracker muon.
 *
 *  \author Luca Ferragina (INFN BO)
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include <vector>

class L1TkMuStubFlatTableProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TkMuStubFlatTableProducer(const edm::ParameterSet &iConfig)
      : src_(consumes<l1t::TrackerMuonCollection>(iConfig.getParameter<edm::InputTag>("src"))),
        name_(iConfig.getParameter<std::string>("name")) {
    produces<nanoaod::FlatTable>();
  }

  void produce(edm::Event &iEvent, const edm::EventSetup &iSetup) override {
    // Input collection
    edm::Handle<l1t::TrackerMuonCollection> muons;
    iEvent.getByToken(src_, muons);

    // Output features
    std::vector<int> muIdxs;
    std::vector<int> etaRegions, phiRegions, depthRegions, types, qualities;

    if (muons.isValid() && !muons->empty()) {
      // Reserve for ~4 stubs per muon
      muIdxs.reserve(muons->size() * 4);
      etaRegions.reserve(muons->size() * 4);
      phiRegions.reserve(muons->size() * 4);
      depthRegions.reserve(muons->size() * 4);
      types.reserve(muons->size() * 4);
      qualities.reserve(muons->size() * 4);

      for (unsigned int iMu = 0; iMu < muons->size(); ++iMu) {
        const auto &mu = muons->at(iMu);
        for (const auto &stubRef : mu.stubs()) {
          if (stubRef.isNull()) {
            continue;
          }

          const auto &stub = *stubRef;
          muIdxs.push_back(iMu);
          etaRegions.push_back(stub.etaRegion());
          phiRegions.push_back(stub.phiRegion());
          depthRegions.push_back(stub.depthRegion());
          types.push_back(stub.type());
          qualities.push_back(stub.quality());
        }
      }
    }

    auto out = std::make_unique<nanoaod::FlatTable>(etaRegions.size(), name_, false);
    out->setDoc("Stubs belonging to L1TkMuons");
    out->addColumn<int>("parentL1TkMu", muIdxs, "Index of the parent L1TkMuon");
    out->addColumn<int>("etaRegion", etaRegions, "Eta region (Wheel/Ring)");
    out->addColumn<int>("phiRegion", phiRegions, "Phi region (Sector/Chamber)");
    out->addColumn<int>("depthRegion", depthRegions, "Station/Depth");
    out->addColumn<int>("type", types, "Stub type: 0 = endcap, 1 = barrel");
    out->addColumn<int>("quality", qualities, "Stub quality");

    iEvent.put(std::move(out));
  }

private:
  edm::EDGetTokenT<l1t::TrackerMuonCollection> src_;
  std::string name_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TkMuStubFlatTableProducer);
