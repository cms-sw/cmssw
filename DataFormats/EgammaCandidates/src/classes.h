#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

namespace {
  namespace {
    reco::PhotonCollection v7;
    edm::Wrapper<reco::PhotonCollection> w7;
    edm::Ref<reco::PhotonCollection> r8;
    edm::RefProd<reco::PhotonCollection> rp8;
    edm::RefVector<reco::PhotonCollection> rv8;

    reco::ElectronCollection v9;
    edm::Wrapper<reco::ElectronCollection> w9;
    edm::Ref<reco::ElectronCollection> r10;
    edm::RefProd<reco::ElectronCollection> rp10;
    edm::RefVector<reco::ElectronCollection> rv10;
  }
}
