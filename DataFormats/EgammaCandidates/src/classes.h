//
// $Id: classes.h,v 1.24 2007/12/08 13:06:11 futyand Exp $
//
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PxPyPzE4D.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h" 
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h" 
#include "FWCore/MessageLogger/interface/MessageLogger.h" 
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/Common/interface/RefProd.h" 
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h" 
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h" 
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h" 
#include "DataFormats/GeometryVector/interface/GlobalPoint.h" 
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h" 
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/CLHEP/interface/Migration.h" 
#include "DataFormats/GeometryVector/interface/LocalPoint.h" 
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h" 
#include "DataFormats/EgammaCandidates/interface/PhotonIsolationAssociation.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronIsoCollection.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronIsoNumCollection.h"
#include "DataFormats/EgammaCandidates/interface/PhotonPi0DiscriminatorAssociation.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCandidateAssociation.h"

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

    reco::GsfElectronCollection v4;
    edm::Wrapper<reco::GsfElectronCollection> w4;
    edm::Ref<reco::GsfElectronCollection> r4;
    edm::RefProd<reco::GsfElectronCollection> rp4;
    edm::Wrapper<edm::RefVector<reco::GsfElectronCollection> > rv4;

    reco::SiStripElectronCollection v5;
    edm::Wrapper<reco::SiStripElectronCollection> w5;
    edm::Ref<reco::SiStripElectronCollection> r5;
    edm::RefProd<reco::SiStripElectronCollection> rp5;
    edm::Wrapper<edm::RefVector<reco::SiStripElectronCollection> > rv5;


    reco::ConversionCollection v6;
    edm::Wrapper<reco::ConversionCollection> w6;
    edm::Ref<reco::ConversionCollection> r6;
    edm::RefProd<reco::ConversionCollection> rp6;
    edm::Wrapper<edm::RefVector<reco::ConversionCollection> > rv6;



    reco::PhotonIsolationMap v66;
    edm::Wrapper<reco::PhotonIsolationMap> w66;
    edm::helpers::Key<edm::RefProd<reco::PhotonCollection > > h66;

    reco::ElectronIsolationMap v7;
    edm::Wrapper<reco::ElectronIsolationMap> w7;
    edm::helpers::Key<edm::RefProd<reco::ElectronCollection > > h7;
    reco::GsfElectronIsoCollection v8;
    reco::GsfElectronIsoCollectionRef r8;
    reco::GsfElectronIsoCollectionRefProd rp8;
    reco::GsfElectronIsoCollectionRefVector rv8;

    edm::Wrapper<reco::GsfElectronIsoCollection> w8;

    reco::GsfElectronIsoNumCollection v9;
    reco::GsfElectronIsoNumCollectionRef r9;
    reco::GsfElectronIsoNumCollectionRefProd rp9;
    reco::GsfElectronIsoNumCollectionRefVector rv9;

    edm::Wrapper<reco::GsfElectronIsoNumCollection> w9;

    reco::PhotonPi0DiscriminatorAssociationMap v10;
    edm::Wrapper<reco::PhotonPi0DiscriminatorAssociationMap> w10;
    edm::helpers::Key<edm::RefProd<reco::PhotonCollection > > h10;

    edm::reftobase::Holder<reco::Candidate, reco::ElectronRef> rb1;
    edm::reftobase::Holder<reco::Candidate, reco::PhotonRef> rb2;
    edm::reftobase::Holder<reco::Candidate, reco::SiStripElectronRef> rb3;


    edm::reftobase::Holder<reco::Candidate, reco::GsfElectronRef> rb11;
    edm::reftobase::RefHolder<reco::GsfElectronRef> rb12;
    edm::reftobase::VectorHolder<reco::Candidate, reco::GsfElectronRefVector> rb13;
    edm::reftobase::RefVectorHolder<reco::GsfElectronRefVector> rb14;

    edm::reftobase::Holder<reco::Candidate, reco::PhotonRef> rb21;
    edm::reftobase::RefHolder<reco::PhotonRef> rb22;
    edm::reftobase::VectorHolder<reco::Candidate, reco::PhotonRefVector> rb23;
    edm::reftobase::RefVectorHolder<reco::PhotonRefVector> rb24;

    edm::Wrapper<reco::PhotonCandidateAssociation> pca1;
  }
}
