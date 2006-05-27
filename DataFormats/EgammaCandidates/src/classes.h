#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ElectronCandidate.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronCandidate.h"
#include "DataFormats/Common/interface/RefToBase.h"

namespace {
  namespace {
    reco::PhotonCollection v1;
    edm::Wrapper<reco::PhotonCollection> w1;
    edm::Ref<reco::PhotonCollection> r1;
    edm::RefProd<reco::PhotonCollection> rp1;
    edm::RefVector<reco::PhotonCollection> rv1;

    reco::ElectronCollection v2;
    edm::Wrapper<reco::ElectronCollection> w2;
    edm::Ref<reco::ElectronCollection> r2;
    edm::RefProd<reco::ElectronCollection> rp2;
    edm::RefVector<reco::ElectronCollection> rv2;

    reco::PhotonCandidateCollection v3;
    edm::Wrapper<reco::PhotonCandidateCollection> w3;
    edm::Ref<reco::PhotonCandidateCollection> r3;
    edm::RefProd<reco::PhotonCandidateCollection> rp3;
    edm::RefVector<reco::PhotonCandidateCollection> rv3;

    reco::ElectronCandidateCollection v4;
    edm::Wrapper<reco::ElectronCandidateCollection> w4;
    edm::Ref<reco::ElectronCandidateCollection> r4;
    edm::RefProd<reco::ElectronCandidateCollection> rp4;
    edm::RefVector<reco::ElectronCandidateCollection> rv4;

    reco::SiStripElectronCandidateCollection v5;
    edm::Wrapper<reco::SiStripElectronCandidateCollection> w5;
    edm::Ref<reco::SiStripElectronCandidateCollection> r5;
    edm::RefProd<reco::SiStripElectronCandidateCollection> rp5;
    edm::RefVector<reco::SiStripElectronCandidateCollection> rv5;

    edm::RefToBaseImpl<reco::Candidate, reco::ElectronCandidateRef> rb1;
    edm::RefToBaseImpl<reco::Candidate, reco::PhotonCandidateRef> rb2;
    edm::RefToBaseImpl<reco::Candidate, reco::SiStripElectronCandidateRef> rb3;
  }
}
