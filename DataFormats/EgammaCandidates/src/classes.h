//
// $Id: classes.h,v 1.41 2011/11/02 17:39:54 nancy Exp $
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonIsolationAssociation.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronIsoCollection.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronIsoNumCollection.h"
#include "DataFormats/EgammaCandidates/interface/PhotonPi0DiscriminatorAssociation.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCandidateAssociation.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/CLHEP/interface/Migration.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "Rtypes.h"
#include "Math/Cartesian3D.h"
#include "Math/Polar3D.h"
#include "Math/CylindricalEta3D.h"
#include "Math/PxPyPzE4D.h"
#include <boost/cstdint.hpp>


namespace {

  struct dictionary {

	  reco::PhotonCoreCollection v0;
    edm::Wrapper<reco::PhotonCoreCollection> w0;
    edm::Ref<reco::PhotonCoreCollection> r0;
    edm::RefProd<reco::PhotonCoreCollection> rp0;
    edm::Wrapper<edm::RefVector<reco::PhotonCoreCollection> > rv0;
    edm::RefToBase<reco::PhotonCore> rtbgpc;
    edm::reftobase::IndirectHolder<reco::PhotonCore> ihgpc;
    edm::RefToBaseProd<reco::PhotonCore> rtbpgpc;
    edm::RefToBaseVector<reco::PhotonCore> rtbvgpc;
    edm::Wrapper<edm::RefToBaseVector<reco::PhotonCore> > rtbvgpc_w;
    edm::reftobase::BaseVectorHolder<reco::PhotonCore> *bvhgpc_p;


    reco::PhotonCollection v1;
    edm::Wrapper<reco::PhotonCollection> w1;
    edm::Ref<reco::PhotonCollection> r1;
    edm::RefProd<reco::PhotonCollection> rp1;
    edm::Wrapper<edm::RefVector<reco::PhotonCollection> > rv1;
    edm::RefToBase<reco::Photon> rtbp;
    edm::reftobase::IndirectHolder<reco::Photon> ihp;
    edm::RefToBaseProd<reco::Photon> rtbpp;
    edm::RefToBaseVector<reco::Photon> rtbvp;
    edm::Wrapper<edm::RefToBaseVector<reco::Photon> > rtbvp_w;
    edm::reftobase::BaseVectorHolder<reco::Photon> *bvhp_p;


    reco::Photon::FiducialFlags pff ;
    reco::Photon::ShowerShape pss ;
    reco::Photon::IsolationVariables piv ;
    reco::Photon::PflowIsolationVariables ppfiv ;
    reco::Photon::PflowIDVariables ppfid ;
    reco::Photon::MIPVariables pmv ;
    reco::Photon::EnergyCorrections pec ;


    reco::ElectronCollection v2;
    edm::Wrapper<reco::ElectronCollection> w2;
    edm::Ref<reco::ElectronCollection> r2;
    edm::RefProd<reco::ElectronCollection> rp2;
    edm::Wrapper<edm::RefVector<reco::ElectronCollection> > rv2;

    edm::RefToBase<reco::Electron> rtbe;
    edm::reftobase::IndirectHolder<reco::Electron> ihe;
    edm::RefToBaseProd<reco::Electron> rtbpe;
    edm::RefToBaseVector<reco::Electron> rtbve;
    edm::Wrapper<edm::RefToBaseVector<reco::Electron> > rtbve_w;
    edm::reftobase::BaseVectorHolder<reco::Electron> *bvhe_p;


    edm::RefToBase<reco::GsfElectronCore> rtbgec;
    edm::reftobase::IndirectHolder<reco::GsfElectronCore> ihgec;
    edm::RefToBaseProd<reco::GsfElectronCore> rtbpgec;
    edm::RefToBaseVector<reco::GsfElectronCore> rtbvgec;
    edm::Wrapper<edm::RefToBaseVector<reco::GsfElectronCore> > rtbvgec_w;
    edm::reftobase::BaseVectorHolder<reco::GsfElectronCore> *bvhgec_p;
    reco::GsfElectronCoreCollection gecc;
    edm::Wrapper<reco::GsfElectronCoreCollection> gecc_w;
    edm::Ref<reco::GsfElectronCoreCollection> gecc_r;
    edm::RefProd<reco::GsfElectronCoreCollection> gecc_rp;
    edm::Wrapper<edm::RefVector<reco::GsfElectronCoreCollection> > gecc_rv;

    reco::GsfElectron::TrackClusterMatching getcm ;
    reco::GsfElectron::TrackExtrapolations gete ;
    reco::GsfElectron::ClosestCtfTrack gecct ;
    reco::GsfElectron::FiducialFlags geff ;
    reco::GsfElectron::ShowerShape gess ;
    reco::GsfElectron::IsolationVariables geiv ;
    reco::GsfElectron::ConversionRejection gecr ;
    reco::GsfElectron::Corrections gec ;
    reco::GsfElectron::ChargeInfo geci ;
    reco::GsfElectron::PflowIsolationVariables gepiv ;
    reco::GsfElectron::MvaInput gemi ;
    reco::GsfElectron::MvaOutput gemo ;
    reco::GsfElectron::ClassificationVariables gecv ;


    edm::RefToBase<reco::GsfElectron> rtbg;
    edm::reftobase::IndirectHolder<reco::GsfElectron> ihg;
    edm::RefToBaseProd<reco::GsfElectron> rtbpg;
    edm::RefToBaseVector<reco::GsfElectron> rtbvg;
    edm::Wrapper<edm::RefToBaseVector<reco::GsfElectron> > rtbvg_w;
    edm::reftobase::BaseVectorHolder<reco::GsfElectron> *bvhg_p;
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

   } ;

  struct ptr
   {
    edm::Ptr<reco::GsfElectron>                         p_gsfElectron ;
    edm::Wrapper<edm::Ptr<reco::GsfElectron> >          w_p_gsfElectron ;

    edm::PtrVector<reco::GsfElectron>                   pv_gsfElectron ;
    edm::Wrapper<edm::PtrVector<reco::GsfElectron> >    w_pv_gsfElectron ;

    edm::Ptr<reco::Photon>	 ptr_ph;
    edm::PtrVector<reco::Photon>	 ptrv_ph;
   } ;

 }
