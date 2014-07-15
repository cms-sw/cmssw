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
#include "DataFormats/MuonReco/interface/MuonPFIsolation.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraFwd.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/MuonReco/interface/MuonQuality.h"
#include "DataFormats/MuonReco/interface/MuonCosmicCompatibility.h"
#include "DataFormats/MuonReco/interface/MuonShower.h"
#include "DataFormats/MuonReco/interface/MuonToMuonMap.h"
#include "DataFormats/TrackReco/interface/Track.h" 
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/MuonReco/interface/DYTInfo.h"

#include <vector>
#include <map>

namespace DataFormats_MuonReco {
  struct dictionary {
    std::vector<reco::Muon> v1;
    edm::Wrapper<std::vector<reco::Muon> > c1;
    edm::Ref<std::vector<reco::Muon> > r1;
    edm::RefProd<std::vector<reco::Muon> > rp1;
    edm::Wrapper<edm::RefVector<std::vector<reco::Muon> > > wrv1;
    edm::reftobase::Holder<reco::Candidate, reco::MuonRef> rb1;
    edm::helpers::Key<edm::RefProd<std::vector<reco::Muon> > > hkrv1;

    reco::MuonIsolation rmi;
    reco::MuonPFIsolation rmi2;
    reco::MuonTime rmt;
    reco::MuonTimeExtra rmt1;
    
    reco::Muon::MuonTrackType rmmttype;
    reco::Muon::MuonTrackRefMap rmmrrmap;

    std::vector<reco::MuonTimeExtra> rmt2;
//    edm::RefProd<std::vector<reco::MuonTimeExtra> > rmt3;
    edm::Wrapper<std::vector<reco::MuonTimeExtra> > wrmt2;
//    edm::helpers::Key<edm::RefProd<std::vector<reco::MuonTimeExtra> > > rmt4;

    reco::MuonTimeExtraMap rmtm;
    edm::Wrapper<reco::MuonTimeExtraMap> wrmtm;

    std::vector<reco::MuonChamberMatch> vmm1;
    std::vector<reco::MuonSegmentMatch> vmm2;
    std::vector<reco::MuonRPCHitMatch>  vmm3;

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
    

    reco::MuonQuality rmq;
    std::vector<reco::MuonQuality> rmq_v;
    std::vector<reco::MuonQuality>::const_iterator rmq_vci;
    edm::Wrapper<std::vector<reco::MuonQuality> > rmq_wv;
    edm::ValueMap<reco::MuonQuality> rmq_vm;
    edm::ValueMap<reco::MuonQuality>::const_iterator rmq_vmci;
    edm::Wrapper<edm::ValueMap<reco::MuonQuality> > rmq_wvm;

    reco::MuonCosmicCompatibility rmcc;
    std::vector<reco::MuonCosmicCompatibility> rmcc_v;
    std::vector<reco::MuonCosmicCompatibility>::const_iterator rmcc_vci;
    edm::Wrapper<std::vector<reco::MuonCosmicCompatibility> > rmcc_wv;
    edm::ValueMap<reco::MuonCosmicCompatibility> rmcc_vm;
    edm::ValueMap<reco::MuonCosmicCompatibility>::const_iterator rmcc_vmci;
    edm::Wrapper<edm::ValueMap<reco::MuonCosmicCompatibility> > rmcc_wvm;


    edm::ValueMap<reco::MuonRef> rmref_vm;
    edm::ValueMap<reco::MuonRef>::const_iterator rmref_vmci;
    edm::Wrapper<edm::ValueMap<reco::MuonRef> > rmref_wvm;


    //shower block
    reco::MuonShower rms;
    std::vector<reco::MuonShower> rms_v;
    std::vector<reco::MuonShower>::const_iterator rms_vci;
    edm::Wrapper<std::vector<reco::MuonShower> > rms_wv;
    edm::ValueMap<reco::MuonShower> rms_vm;
    edm::ValueMap<reco::MuonShower>::const_iterator rms_vmci;
    edm::Wrapper<edm::ValueMap<reco::MuonShower> > rms_wvm;

    // DYT part
    reco::DYTInfo rdyt;
    std::vector<reco::DYTInfo> rdyt_v;
    std::vector<reco::DYTInfo>::const_iterator rdyt_vci;
    edm::Wrapper<std::vector<reco::DYTInfo> > rdyt_wv;
    edm::ValueMap<reco::DYTInfo> rdyt_vm;
    edm::ValueMap<reco::DYTInfo>::const_iterator rdyt_vmci;
    edm::Wrapper<edm::ValueMap<reco::DYTInfo> > rdyt_wvm;
    
    //Ptrs
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
    edm::Ptr<reco::Muon>                         p_muon;
    edm::Wrapper<edm::Ptr<reco::Muon> >          w_p_muon;

    edm::PtrVector<reco::Muon>                   pv_muon;
    edm::Wrapper<edm::PtrVector<reco::Muon> >    w_pv_muon;

  };
}

