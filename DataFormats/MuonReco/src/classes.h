#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"
#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PxPyPzE4D.h" 
#include <boost/cstdint.hpp> 
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/Common/interface/AssociationMap.h"

#define USE_MUISODEPOSIT_REQUIRED
#include "DataFormats/MuonReco/interface/Direction.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h" 

#include <vector>
#include <map>

namespace {
  struct dictionary {
    std::vector<reco::Muon> v1;
    edm::Wrapper<std::vector<reco::Muon> > c1;
    edm::Ref<std::vector<reco::Muon> > r1;
    edm::RefProd<std::vector<reco::Muon> > rp1;
    edm::Wrapper<edm::RefVector<std::vector<reco::Muon> > > wrv1;
    edm::reftobase::Holder<reco::Candidate, reco::MuonRef> rb1;
    edm::helpers::Key<edm::RefProd<std::vector<reco::Muon> > > hkrv1;

    std::multimap<muonisolation::Direction::Distance,float> v2b;
    reco::MuIsoDeposit miso;
    std::vector<reco::MuIsoDeposit> v2;
    edm::Wrapper<std::vector<reco::MuIsoDeposit> > c2;
    edm::Ref<std::vector<reco::MuIsoDeposit> > r2;
    edm::RefProd<std::vector<reco::MuIsoDeposit> > rp2;
    edm::Wrapper<edm::RefVector<std::vector<reco::MuIsoDeposit> > > wrv2;
    
    reco::MuonIsolation rmi;
    reco::MuonTime rmt;

    std::vector<reco::MuonChamberMatch> vmm1;
    std::vector<reco::MuonSegmentMatch> vmm2;

//defined in DataFormats/TrackReco
//    reco::MuIsoAssociationMap v4;
//    edm::Wrapper<reco::MuIsoAssociationMap> w4;

    std::map<unsigned int, reco::MuIsoDeposit> m5;
    reco::MuIsoDepositAssociationMap v5;
    edm::Wrapper<reco::MuIsoDepositAssociationMap> w5;

//defined in DataFormats/TrackReco
//    reco::MuIsoIntAssociationMap v6;
//    edm::Wrapper<reco::MuIsoIntAssociationMap> w6;

//    reco::MuIsoFloatAssociationMap v7;
//    edm::Wrapper<reco::MuIsoFloatAssociationMap> w7;
   

   
    reco::MuIsoAssociationMapToMuon v8;
    edm::Wrapper<reco::MuIsoAssociationMapToMuon> w8;

    reco::MuIsoDepositAssociationMapToMuon v9;
    edm::Wrapper<reco::MuIsoDepositAssociationMapToMuon> w9;

    reco::MuIsoIntAssociationMapToMuon v91;
    edm::Wrapper<reco::MuIsoIntAssociationMapToMuon> w91;

    reco::MuIsoFloatAssociationMapToMuon v10;
    edm::Wrapper<reco::MuIsoFloatAssociationMapToMuon> w10;
   

    reco::MuIsoAssociationVector v11;
    edm::Wrapper<reco::MuIsoAssociationVector> w11;

    reco::MuIsoDepositAssociationVector v12;
    edm::Wrapper<reco::MuIsoDepositAssociationVector> w12;

    //the two below will have to be in DataFormats/TrackReco at some point
    reco::MuIsoIntAssociationVector v13;
    edm::Wrapper<reco::MuIsoIntAssociationVector> w13;

    reco::MuIsoFloatAssociationVector v14;
    edm::Wrapper<reco::MuIsoFloatAssociationVector> w14;
   

    reco::MuIsoAssociationVectorToMuon v19;
    edm::Wrapper<reco::MuIsoAssociationVectorToMuon> w19;

    reco::MuIsoDepositAssociationVectorToMuon v20;
    edm::Wrapper<reco::MuIsoDepositAssociationVectorToMuon> w20;

    reco::MuIsoIntAssociationVectorToMuon v21;
    edm::Wrapper<reco::MuIsoIntAssociationVectorToMuon> w21;

    reco::MuIsoFloatAssociationVectorToMuon v22;
    edm::Wrapper<reco::MuIsoFloatAssociationVectorToMuon> w22;
   

    reco::MuIsoDepositAssociationVectorToCandidateView v23;
    edm::Wrapper<reco::MuIsoDepositAssociationVectorToCandidateView> w23;


    std::vector<reco::MuonTrackLinks> tl1;
    edm::Wrapper<std::vector<reco::MuonTrackLinks> > tl2;
    edm::Ref<std::vector<reco::MuonTrackLinks> > tl3;
    edm::RefProd<std::vector<reco::MuonTrackLinks> > tl4;
    edm::Wrapper<edm::RefVector<std::vector<reco::MuonTrackLinks> > > wtl5;

    std::vector<reco::CaloMuon> smv1;
    edm::Wrapper<std::vector<reco::CaloMuon> > smc1;

    edm::reftobase::Holder<reco::Candidate, reco::MuonRef> hcc1;
    edm::reftobase::RefHolder<reco::MuonRef> hcc2;
    edm::reftobase::VectorHolder<reco::Candidate, reco::MuonRefVector> hcc3;
    edm::reftobase::RefVectorHolder<reco::MuonRefVector> hcc4;

    reco::CandIsoDepositAssociationPair candIsoAP;
    reco::CandIsoDepositAssociationVector candIsoAV;
    std::vector<reco::CandIsoDepositAssociationVector> candIsoAPV_root_seems_to_need_this;
    edm::Wrapper<reco::CandIsoDepositAssociationVector> candIsoAV_w;

    reco::MuIsoDepositMap idvm;
    reco::MuIsoDepositMap::const_iterator idvmci;
    edm::Wrapper<reco::MuIsoDepositMap> w_idvm;
  };
}

