#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/ValueMap.h"
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
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraFwd.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/Common/interface/AssociationMap.h"


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

    reco::MuonIsolation rmi;
    reco::MuonTime rmt;
    reco::MuonTimeExtra rmt1;
    
    std::vector<reco::MuonTimeExtra> rmt2;
//    edm::RefProd<std::vector<reco::MuonTimeExtra> > rmt3;
    edm::Wrapper<std::vector<reco::MuonTimeExtra> > wrmt2;
//    edm::helpers::Key<edm::RefProd<std::vector<reco::MuonTimeExtra> > > rmt4;

    reco::MuonTimeExtraMap rmtm;
    edm::Wrapper<reco::MuonTimeExtraMap> wrmtm;

    std::vector<reco::MuonChamberMatch> vmm1;
    std::vector<reco::MuonSegmentMatch> vmm2;

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

    edm::RefToBase<reco::Muon> rtbm;
    edm::reftobase::IndirectHolder<reco::Muon> ihm;
    edm::RefToBaseProd<reco::Muon> rtbpm;
    edm::RefToBaseVector<reco::Muon> rtbvm;
    edm::Wrapper<edm::RefToBaseVector<reco::Muon> > rtbvm_w;
    edm::reftobase::BaseVectorHolder<reco::Muon> *bvhm_p;

    reco::MuonMETCorrectionData rmcd;
    std::vector<reco::MuonMETCorrectionData> rmcd_v;
    std::vector<reco::MuonMETCorrectionData>::const_iterator rmcd_vci;
    edm::Wrapper<std::vector<reco::MuonMETCorrectionData> > rmcd_wv;
    edm::ValueMap<reco::MuonMETCorrectionData> rmcd_vm;
    edm::ValueMap<reco::MuonMETCorrectionData>::const_iterator rmcd_vmci;
    edm::Wrapper<edm::ValueMap<reco::MuonMETCorrectionData> > rmcd_wvm;
    
  };
}

