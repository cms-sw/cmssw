//
// EGMSeedLocationProducer (to compute iEta/iPhi or iX/iY of seed)
//
// Author: Swagata Mukherjee
// Date: December 2022
//

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

template <typename T>
class EGMSeedLocationProducer : public edm::global::EDProducer<> {
public:
  explicit EGMSeedLocationProducer(const edm::ParameterSet& iConfig)
      : src_(consumes<edm::View<T>>(iConfig.getParameter<edm::InputTag>("src"))) {
    //recHitsEB_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsEB"))),
    //recHitsEE_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("recHitsEE"))) {
    produces<edm::ValueMap<int>>("iEta");
    produces<edm::ValueMap<int>>("iPhi");
    produces<edm::ValueMap<int>>("iX");
    produces<edm::ValueMap<int>>("iY");
    //produces<edm::ValueMap<float>>("PFIsoChg");
  }
  ~EGMSeedLocationProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<T>> src_;
  //  edm::EDGetTokenT<EcalRecHitCollection> recHitsEB_;
  // edm::EDGetTokenT<EcalRecHitCollection> recHitsEE_;
};

template <typename T>
void EGMSeedLocationProducer<T>::produce(edm::StreamID streamID,
                                         edm::Event& iEvent,
                                         const edm::EventSetup& iSetup) const {
  auto src = iEvent.getHandle(src_);
  //  const auto& recHitsEBProd = iEvent.get(recHitsEB_);
  //const auto& recHitsEEProd = iEvent.get(recHitsEE_);

  unsigned nSrc = src->size();
  //  std::vector<int> gainSeed(nSrc);
  std::vector<int> iEta(nSrc, 0);
  std::vector<int> iPhi(nSrc, 0);
  std::vector<int> iX(nSrc, 0);
  std::vector<int> iY(nSrc, 0);

  for (unsigned i = 0; i < nSrc; i++) {  // object loop
    auto obj = src->ptrAt(i);
    auto detid = obj->superCluster()->seed()->seed();

    if (detid.subdetId() == EcalBarrel) {
      EBDetId ebdetid(detid);
      iEta[i] = ebdetid.ieta();
      iPhi[i] = ebdetid.iphi();
    } else if (detid.subdetId() == EcalEndcap) {
      EEDetId eedetid(detid);
      iX[i] = eedetid.ix();
      iY[i] = eedetid.iy();
    }
  }  // end of object loop

  std::unique_ptr<edm::ValueMap<int>> iEtaV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filleriEtaV(*iEtaV);
  filleriEtaV.insert(src, iEta.begin(), iEta.end());
  filleriEtaV.fill();
  iEvent.put(std::move(iEtaV), "iEta");
  //  iEvent.put(std::move(PFIsoChgV), "PFIsoChg");

  std::unique_ptr<edm::ValueMap<int>> iPhiV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filleriPhiV(*iPhiV);
  filleriPhiV.insert(src, iPhi.begin(), iPhi.end());
  filleriPhiV.fill();
  iEvent.put(std::move(iPhiV), "iPhi");

  std::unique_ptr<edm::ValueMap<int>> iXV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filleriXV(*iXV);
  filleriXV.insert(src, iX.begin(), iX.end());
  filleriXV.fill();
  iEvent.put(std::move(iXV), "iX");

  std::unique_ptr<edm::ValueMap<int>> iYV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filleriYV(*iYV);
  filleriYV.insert(src, iY.begin(), iY.end());
  filleriYV.fill();
  iEvent.put(std::move(iYV), "iY");

  //  std::unique_ptr<edm::ValueMap<int>> gainSeedV(new edm::ValueMap<int>());
  // edm::ValueMap<int>::Filler fillerCorr(*gainSeedV);
  //  fillerCorr.insert(src, gainSeed.begin(), gainSeed.end());
  // fillerCorr.fill();
  // iEvent.put(std::move(gainSeedV));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
template <typename T>
void EGMSeedLocationProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  //  desc.add<edm::InputTag>("recHitsEB", edm::InputTag("reducedEgamma", "reducedEBRecHits"))
  //  ->setComment("EB rechit collection");
  //desc.add<edm::InputTag>("recHitsEE", edm::InputTag("reducedEgamma", "reducedEERecHits"))
  //  ->setComment("EE rechit collection");
  descriptions.addDefault(desc);
}

typedef EGMSeedLocationProducer<pat::Electron> ElectronSeedLocationProducer;
typedef EGMSeedLocationProducer<pat::Photon> PhotonSeedLocationProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronSeedLocationProducer);
DEFINE_FWK_MODULE(PhotonSeedLocationProducer);
