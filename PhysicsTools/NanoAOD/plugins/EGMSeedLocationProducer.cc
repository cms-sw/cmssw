//
// EGMSeedLocationProducer
// to compute iEta/iPhi (for barrel) and iX/iY (for endcaps) of seed crystal
// for size considerations, they are compressed into 2 vars (iEtaOriX/iPhiOriY)
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
    produces<edm::ValueMap<int>>("iEtaOriX");
    produces<edm::ValueMap<int>>("iPhiOriY");
  }
  ~EGMSeedLocationProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<edm::View<T>> src_;
};

template <typename T>
void EGMSeedLocationProducer<T>::produce(edm::StreamID streamID,
                                         edm::Event& iEvent,
                                         const edm::EventSetup& iSetup) const {
  auto src = iEvent.getHandle(src_);

  unsigned nSrc = src->size();
  // Range of the variables are the following:
  // iEta runs from -85 to +85, with no crystal at iEta=0.
  // iPhi runs from 1 to 360.
  // iX and iY run from 1 to 100.
  // So, when combined, iEtaOriX will be -85 to 100 (except 0).
  // and iPhiOriY will be 1 to 360.
  std::vector<int> iEtaOriX(nSrc, 0);
  std::vector<int> iPhiOriY(nSrc, 0);

  for (unsigned i = 0; i < nSrc; i++) {  // object loop
    auto obj = src->ptrAt(i);
    auto detid = obj->superCluster()->seed()->seed();

    if (detid.subdetId() == EcalBarrel) {
      EBDetId ebdetid(detid);
      iEtaOriX[i] = ebdetid.ieta();
      iPhiOriY[i] = ebdetid.iphi();
    } else if (detid.subdetId() == EcalEndcap) {
      EEDetId eedetid(detid);
      iEtaOriX[i] = eedetid.ix();
      iPhiOriY[i] = eedetid.iy();
    }
  }  // end of object loop

  std::unique_ptr<edm::ValueMap<int>> iEtaOriXV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filleriEtaOriXV(*iEtaOriXV);
  filleriEtaOriXV.insert(src, iEtaOriX.begin(), iEtaOriX.end());
  filleriEtaOriXV.fill();
  iEvent.put(std::move(iEtaOriXV), "iEtaOriX");

  std::unique_ptr<edm::ValueMap<int>> iPhiOriYV(new edm::ValueMap<int>());
  edm::ValueMap<int>::Filler filleriPhiOriYV(*iPhiOriYV);
  filleriPhiOriYV.insert(src, iPhiOriY.begin(), iPhiOriY.end());
  filleriPhiOriYV.fill();
  iEvent.put(std::move(iPhiOriYV), "iPhiOriY");
}

template <typename T>
void EGMSeedLocationProducer<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src")->setComment("input physics object collection");
  descriptions.addDefault(desc);
}

typedef EGMSeedLocationProducer<pat::Electron> ElectronSeedLocationProducer;
typedef EGMSeedLocationProducer<pat::Photon> PhotonSeedLocationProducer;

DEFINE_FWK_MODULE(ElectronSeedLocationProducer);
DEFINE_FWK_MODULE(PhotonSeedLocationProducer);
