//
// $Id: classes.h,v 1.16 2007/06/08 13:47:58 llista Exp $
//
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/ConvertedPhoton.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/PhotonIsolationAssociation.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/AssociationMap.h"


namespace {
  namespace {
    reco::PhotonCollection v1;
    edm::Wrapper<reco::PhotonCollection> w1;
    edm::Ref<reco::PhotonCollection> r1;
    edm::RefProd<reco::PhotonCollection> rp1;
    edm::Wrapper<edm::RefVector<reco::PhotonCollection> > rv1;

    reco::ElectronCollection v2;
    edm::Wrapper<reco::ElectronCollection> w2;
    edm::Ref<reco::ElectronCollection> r2;
    edm::RefProd<reco::ElectronCollection> rp2;
    edm::Wrapper<edm::RefVector<reco::ElectronCollection> > rv2;

    reco::PixelMatchElectronCollection v3;
    edm::Wrapper<reco::PixelMatchElectronCollection> w3;
    edm::Ref<reco::PixelMatchElectronCollection> r3;
    edm::RefProd<reco::PixelMatchElectronCollection> rp3;
    edm::Wrapper<edm::RefVector<reco::PixelMatchElectronCollection> > rv3;

    reco::PixelMatchGsfElectronCollection v4;
    edm::Wrapper<reco::PixelMatchGsfElectronCollection> w4;
    edm::Ref<reco::PixelMatchGsfElectronCollection> r4;
    edm::RefProd<reco::PixelMatchGsfElectronCollection> rp4;
    edm::Wrapper<edm::RefVector<reco::PixelMatchGsfElectronCollection> > rv4;

    reco::SiStripElectronCollection v5;
    edm::Wrapper<reco::SiStripElectronCollection> w5;
    edm::Ref<reco::SiStripElectronCollection> r5;
    edm::RefProd<reco::SiStripElectronCollection> rp5;
    edm::Wrapper<edm::RefVector<reco::SiStripElectronCollection> > rv5;

    reco::ConvertedPhotonCollection v6;
    edm::Wrapper<reco::ConvertedPhotonCollection> w6;
    edm::Ref<reco::ConvertedPhotonCollection> r6;
    edm::RefProd<reco::ConvertedPhotonCollection> rp6;
    edm::Wrapper<edm::RefVector<reco::ConvertedPhotonCollection> > rv6;

    reco::PhotonIsolationMap v66;
    edm::Wrapper<reco::PhotonIsolationMap> w66;
    edm::helpers::Key<edm::RefProd<reco::PhotonCollection > > h66;

    reco::ElectronIsolationMap v7;
    edm::Wrapper<reco::ElectronIsolationMap> w7;
    edm::helpers::Key<edm::RefProd<reco::ElectronCollection > > h7;


    edm::reftobase::Holder<reco::Candidate, reco::ElectronRef> rb1;
    edm::reftobase::Holder<reco::Candidate, reco::PhotonRef> rb2;
    edm::reftobase::Holder<reco::Candidate, reco::SiStripElectronRef> rb3;
    edm::reftobase::Holder<reco::Candidate, reco::ConvertedPhotonRef> rb4;

    edm::reftobase::Holder<reco::Candidate, reco::PixelMatchGsfElectronRef> rb11;
    edm::reftobase::RefHolder<reco::PixelMatchGsfElectronRef> rb12;
    edm::reftobase::VectorHolder<reco::Candidate, reco::PixelMatchGsfElectronRefVector> rb13;
    edm::reftobase::RefVectorHolder<reco::PixelMatchGsfElectronRefVector> rb14;
    
    edm::reftobase::Holder<reco::Candidate, reco::PhotonRef> rb21;
    edm::reftobase::RefHolder<reco::PhotonRef> rb22;
    edm::reftobase::VectorHolder<reco::Candidate, reco::PhotonRefVector> rb23;
    edm::reftobase::RefVectorHolder<reco::PhotonRefVector> rb24;
    
    edm::reftobase::Holder<reco::Candidate, reco::ConvertedPhotonRef> rb31;
    edm::reftobase::RefHolder<reco::ConvertedPhotonRef> rb32;
    edm::reftobase::VectorHolder<reco::Candidate, reco::ConvertedPhotonRefVector> rb33;
    edm::reftobase::RefVectorHolder<reco::ConvertedPhotonRefVector> rb34;
  }
}
