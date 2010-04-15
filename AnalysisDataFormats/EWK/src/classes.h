#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "AnalysisDataFormats/EWK/interface/DiLeptonRefBaseCandidate.h"
#include "AnalysisDataFormats/EWK/interface/LeptonMETRefBaseCandidate.h"
#include "AnalysisDataFormats/EWK/interface/ZGammaRefBaseCandidate.h"
#include "AnalysisDataFormats/EWK/interface/WMuNuCandidatePtr.h"
#include "AnalysisDataFormats/EWK/interface/WMuNuCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Candidate/interface/ShallowClonePtrCandidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"
#include "DataFormats/Common/interface/VectorHolder.h"
#include "DataFormats/Common/interface/BaseVectorHolder.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"



#include <vector>
#include <map>


namespace {
  struct dictionary {
     std::vector<reco::WMuNuCandidate> v1;
     edm::Wrapper<std::vector<reco::WMuNuCandidate> > c1;
     edm::Ref<std::vector<reco::WMuNuCandidate> > r1;
     edm::RefProd<std::vector<reco::WMuNuCandidate> > rp1;
     edm::Wrapper<edm::RefVector<std::vector<reco::WMuNuCandidate> > > wrv1;
     edm::helpers::Key<edm::RefProd<std::vector<reco::WMuNuCandidate> > > hkrv1;
     edm::RefToBase<reco::WMuNuCandidate> rtbm;
     edm::reftobase::IndirectHolder<reco::WMuNuCandidate> ihm;
     edm::RefToBaseProd<reco::WMuNuCandidate> rtbpm;
     edm::RefToBaseVector<reco::WMuNuCandidate> rtbvm;
     edm::Wrapper<edm::RefToBaseVector<reco::WMuNuCandidate> > rtbvm_w;
     edm::reftobase::BaseVectorHolder<reco::WMuNuCandidate> *bvhm_p;
 


     std::vector<reco::WMuNuCandidatePtr> v2;
     edm::Wrapper<std::vector<reco::WMuNuCandidatePtr> > c2;
     edm::Ref<std::vector<reco::WMuNuCandidatePtr> > r2;
     edm::RefProd<std::vector<reco::WMuNuCandidatePtr> > rp2;
     edm::Wrapper<edm::RefVector<std::vector<reco::WMuNuCandidatePtr> > > wrv2;
     edm::helpers::Key<edm::RefProd<std::vector<reco::WMuNuCandidatePtr> > > hkrv2;
     edm::RefToBase<reco::WMuNuCandidatePtr> rtbm2;
     edm::reftobase::IndirectHolder<reco::WMuNuCandidatePtr> ihm2;
     edm::RefToBaseProd<reco::WMuNuCandidatePtr> rtbpm2;
     edm::RefToBaseVector<reco::WMuNuCandidatePtr> rtbvm2;
     edm::Wrapper<edm::RefToBaseVector<reco::WMuNuCandidatePtr> > rtbvm_w2;
     edm::reftobase::BaseVectorHolder<reco::WMuNuCandidatePtr> *bvhm_p2;
    
     std::vector<ewk::DiLeptonRefBaseCandidate> v3;
     edm::Wrapper<std::vector<ewk::DiLeptonRefBaseCandidate> > c3;
     edm::Ref<std::vector<ewk::DiLeptonRefBaseCandidate> > r3;
     edm::RefProd<std::vector<ewk::DiLeptonRefBaseCandidate> > rp3;
     edm::Wrapper<edm::RefVector<std::vector<ewk::DiLeptonRefBaseCandidate> > > wrv3;
     edm::helpers::Key<edm::RefProd<std::vector<ewk::DiLeptonRefBaseCandidate> > > hkrv3;
     edm::RefToBase<ewk::DiLeptonRefBaseCandidate> rtbm3;
     edm::reftobase::IndirectHolder<ewk::DiLeptonRefBaseCandidate> ihm3;
     edm::RefToBaseProd<ewk::DiLeptonRefBaseCandidate> rtbpm3;
     edm::RefToBaseVector<ewk::DiLeptonRefBaseCandidate> rtbvm3;
     edm::Wrapper<edm::RefToBaseVector<ewk::DiLeptonRefBaseCandidate> > rtbvm_w3;
     edm::reftobase::BaseVectorHolder<ewk::DiLeptonRefBaseCandidate> *bvhm_p3;
    
     std::vector<ewk::LeptonMETRefBaseCandidate> v4;
     edm::Wrapper<std::vector<ewk::LeptonMETRefBaseCandidate> > c4;
     edm::Ref<std::vector<ewk::LeptonMETRefBaseCandidate> > r4;
     edm::RefProd<std::vector<ewk::LeptonMETRefBaseCandidate> > rp4;
     edm::Wrapper<edm::RefVector<std::vector<ewk::LeptonMETRefBaseCandidate> > > wrv4;
     edm::helpers::Key<edm::RefProd<std::vector<ewk::LeptonMETRefBaseCandidate> > > hkrv4;
     edm::RefToBase<ewk::LeptonMETRefBaseCandidate> rtbm4;
     edm::reftobase::IndirectHolder<ewk::LeptonMETRefBaseCandidate> ihm4;
     edm::RefToBaseProd<ewk::LeptonMETRefBaseCandidate> rtbpm4;
     edm::RefToBaseVector<ewk::LeptonMETRefBaseCandidate> rtbvm4;
     edm::Wrapper<edm::RefToBaseVector<ewk::LeptonMETRefBaseCandidate> > rtbvm_w4;
     edm::reftobase::BaseVectorHolder<ewk::LeptonMETRefBaseCandidate> *bvhm_p4;

     std::vector<ewk::ZGammaRefBaseCandidate> v5;
     edm::Wrapper<std::vector<ewk::ZGammaRefBaseCandidate> > c5;
     edm::Ref<std::vector<ewk::ZGammaRefBaseCandidate> > r5;
     edm::RefProd<std::vector<ewk::ZGammaRefBaseCandidate> > rp5;
     edm::Wrapper<edm::RefVector<std::vector<ewk::ZGammaRefBaseCandidate> > > wrv5;
     edm::helpers::Key<edm::RefProd<std::vector<ewk::ZGammaRefBaseCandidate> > > hkrv5;
     edm::RefToBase<ewk::ZGammaRefBaseCandidate> rtbm5;
     edm::reftobase::IndirectHolder<ewk::ZGammaRefBaseCandidate> ihm5;
     edm::RefToBaseProd<ewk::ZGammaRefBaseCandidate> rtbpm5;
     edm::RefToBaseVector<ewk::ZGammaRefBaseCandidate> rtbvm5;
     edm::Wrapper<edm::RefToBaseVector<ewk::ZGammaRefBaseCandidate> > rtbvm_w5;
     edm::reftobase::BaseVectorHolder<ewk::ZGammaRefBaseCandidate> *bvhm_p5;
    
  };
}  
