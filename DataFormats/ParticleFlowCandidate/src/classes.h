#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidate.h"

#include "Rtypes.h" 
#include "Math/Cartesian3D.h" 
#include "Math/Polar3D.h" 
#include "Math/CylindricalEta3D.h" 
#include "Math/PxPyPzE4D.h" 
#include "DataFormats/TrackReco/interface/TrackFwd.h" 

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtraFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtraFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PileUpPFCandidateFwd.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/Association.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Common/interface/OneToManyWithQualityGeneric.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/RefVectorHolder.h"

namespace DataFormats_ParticleFlowCandidate {
  struct dictionary {

    reco::PFCandidateRef c_r;
    reco::PFCandidateRefProd c_rp;
    reco::PFCandidateRefVector c_rv;
    edm::Wrapper<std::vector<reco::PFCandidate> > w1;
    edm::Wrapper<reco::PFCandidate> pfcand_w1;
    edm::reftobase::Holder<reco::Candidate, reco::PFCandidateRef> bla1; 
    edm::reftobase::RefHolder<reco::PFCandidateRef> bla2; 
    reco::PFCandidatePtr bla3;     
    std::vector<reco::PFCandidatePtr> bla33;     
    edm::Wrapper<edm::ValueMap<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > > > bla333;
    edm::ValueMap<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > >  bla334;
    std::vector<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > > bla335;
    edm::Wrapper<edm::ValueMap<std::vector<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > > > > valueMap_iso_wr;
    edm::ValueMap<std::vector<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> > > >  valueMap_iso; 
    edm::reftobase::RefVectorHolder<reco::PFCandidateRefVector > bla3351;

    edm::Wrapper<edm::ValueMap<edm::Ptr<reco::PFCandidate> > > bla336;
    edm::ValueMap<edm::Ptr<std::vector<reco::PFCandidate> > >  bla337;
    std::vector<edm::Ptr<std::vector<reco::PFCandidate> > > bla338;
    edm::ValueMap<edm::Ptr<reco::PFCandidate> > bla339;
    reco::PFCandidate::ElementInBlock jo1;
    reco::PFCandidate::ElementsInBlocks jo2;  

    edm::PtrVector<reco::PFCandidate> mm1;
    edm::Wrapper<edm::PtrVector<reco::PFCandidate> > mm2;

    reco::IsolatedPFCandidateRef ic_r;
    reco::IsolatedPFCandidateRefProd ic_rp;
    reco::IsolatedPFCandidateRefVector ic_rv;
    edm::Wrapper<std::vector<reco::IsolatedPFCandidate> > iw1;
    reco::IsolatedPFCandidatePtr bla4;     

    reco::PileUpPFCandidateRef puc_r;
    reco::PileUpPFCandidateRefProd puc_rp;
    reco::PileUpPFCandidateRefVector puc_rv;
    edm::Wrapper<std::vector<reco::PileUpPFCandidate> > puw1;
    reco::PileUpPFCandidatePtr bla5;     

    reco::PFCandidateFwdRef c_fr;
    edm::reftobase::Holder<reco::Candidate, reco::PFCandidateFwdRef> bla1f; 
    edm::reftobase::RefHolder<reco::PFCandidateFwdRef> bla2f; 
    reco::PFCandidateFwdPtr fbla3;     
    std::vector<reco::PFCandidateFwdRef> bla33f;     
    std::vector<reco::PFCandidateFwdPtr> bla33fv; 
    edm::Wrapper<std::vector<reco::PFCandidateFwdPtr > > bla33fvw;
    
    reco::PFCandidateElectronExtraRef ex_r;
    reco::PFCandidateElectronExtraRefProd ex_rp;
    reco::PFCandidateElectronExtraRefVector ex_rv;
    edm::Wrapper<std::vector<reco::PFCandidateElectronExtra> > ex_w1;

    reco::PFCandidatePhotonExtraRef ph_r;
    reco::PFCandidatePhotonExtraRefProd ph_rp;
    //    reco::PFCandidatePhotonExtraRefVector ph_rv;
    edm::Wrapper<std::vector<reco::PFCandidatePhotonExtra> > ph_w1;

    reco::PFCandidateEGammaExtraRef eg_r;
    reco::PFCandidateEGammaExtraRefProd eg_rp;
    reco::PFCandidateEGammaExtraRefVector eg_rv;
    edm::Wrapper<reco::PFCandidateEGammaExtra> eg_cand_w1;  
    edm::Wrapper<std::vector<reco::PFCandidateEGammaExtra> > eg_w1;    

    edm::Wrapper<edm::Association<reco::PFCandidateCollection > >    w_asso_pfc;
 
    //Association Map classes
    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::Vertex> >,edm::RefProd<std::vector<reco::PFCandidate> > > pm0;
    edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,float> > > pmf1;
    edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::PFCandidate>,float,unsigned int> > pmf2;
    edm::Wrapper<edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::PFCandidate>,float,unsigned int> > > pmf3;
    std::map<unsigned int,edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,float> > > > pmf4;
    std::vector<std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,float> > pmf5;
    std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,float> pmf6;
    edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,int> > > pm1;
    edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::PFCandidate>,int,unsigned int> > pm2;
    edm::Wrapper<edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::Vertex>,std::vector<reco::PFCandidate>,int,unsigned int> > > pm3;
    std::map<unsigned int,edm::helpers::KeyVal<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,std::vector<std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,int> > > > pm4;
    std::vector<std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,int> > pm5;
    std::pair<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,int> pm6;


    edm::helpers::KeyVal<edm::RefProd<std::vector<reco::PFCandidate> >,edm::RefProd<std::vector<reco::Vertex> > > mp0;
    edm::helpers::KeyVal<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,std::vector<std::pair<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,int> > > mp1;
    edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::PFCandidate>,std::vector<reco::Vertex>,int,unsigned int> > mp2;
    edm::Wrapper<edm::AssociationMap<edm::OneToManyWithQuality<std::vector<reco::PFCandidate>,std::vector<reco::Vertex>,int,unsigned int> > > mp3;
    std::map<unsigned int,edm::helpers::KeyVal<edm::Ref<std::vector<reco::PFCandidate>,reco::PFCandidate,edm::refhelper::FindUsingAdvance<std::vector<reco::PFCandidate>,reco::PFCandidate> >,std::vector<std::pair<edm::Ref<std::vector<reco::Vertex>,reco::Vertex,edm::refhelper::FindUsingAdvance<std::vector<reco::Vertex>,reco::Vertex> >,int> > > > mp4;

  };
}
