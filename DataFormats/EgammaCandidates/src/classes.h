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
#include "DataFormats/EgammaCandidates/interface/PMGsfElectronIsoCollection.h"
#include "DataFormats/EgammaCandidates/interface/PMGsfElectronIsoNumCollection.h"
#include "DataFormats/EgammaCandidates/interface/PhotonPi0DiscriminatorAssociation.h"
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

    reco::PMGsfElectronIsoCollectionBase b8;
    reco::PMGsfElectronIsoCollection v8;
    reco::PMGsfElectronIsoCollectionRef r8;
    reco::PMGsfElectronIsoCollectionRefProd rp8;
    reco::PMGsfElectronIsoCollectionRefVector rv8;
    
    edm::Wrapper<reco::PMGsfElectronIsoCollection> w8;

    reco::PMGsfElectronIsoNumCollectionBase b9;
    reco::PMGsfElectronIsoNumCollection v9;
    reco::PMGsfElectronIsoNumCollectionRef r9;
    reco::PMGsfElectronIsoNumCollectionRefProd rp9;
    reco::PMGsfElectronIsoNumCollectionRefVector rv9;
    
    edm::Wrapper<reco::PMGsfElectronIsoNumCollection> w9;

    reco::PhotonPi0DiscriminatorAssociationMap v10;
    edm::Wrapper<reco::PhotonPi0DiscriminatorAssociationMap> w10;
    edm::helpers::Key<edm::RefProd<reco::PhotonCollection > > h10;

    edm::reftobase::Holder<reco::Candidate, reco::ElectronRef> rb1;
    edm::reftobase::Holder<reco::Candidate, reco::PhotonRef> rb2;
    edm::reftobase::Holder<reco::Candidate, reco::SiStripElectronRef> rb3;
    edm::reftobase::Holder<reco::Candidate, reco::ConvertedPhotonRef> rb4;
  }
}
