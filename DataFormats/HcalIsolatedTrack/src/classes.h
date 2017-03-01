#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/HcalIsolatedTrack/interface/HcalIsolatedTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidate.h"
namespace DataFormats_HcalIsolatedTrack {
  struct dictionary {
    edm::Wrapper<std::map< int, std::pair<double,double> > > w2;
 

    reco::IsolatedPixelTrackCandidate                                             ptc1;
    reco::IsolatedPixelTrackCandidateCollection                                   ptc_c1;
    reco::IsolatedPixelTrackCandidateRef                                          ptc_r1;
    reco::IsolatedPixelTrackCandidateRefProd                                      ptc_rp1;
    reco::IsolatedPixelTrackCandidateRefVector                                    ptc_rv1;
    edm::Wrapper<reco::IsolatedPixelTrackCandidateCollection>                     ptc_wc1;
    edm::reftobase::Holder<reco::Candidate, reco::IsolatedPixelTrackCandidateRef> ptc_h1;
    edm::Ref<reco::IsolatedPixelTrackCandidateCollection> 		          ptc_sr1;
    reco::IsolatedPixelTrackCandidateSimpleRefVector 			          ptc_srv1;

    reco::EcalIsolatedParticleCandidate                                           ptc2;
    reco::EcalIsolatedParticleCandidateCollection                                 ptc_c2;
    reco::EcalIsolatedParticleCandidateRef                                        ptc_r2;
    reco::EcalIsolatedParticleCandidateRefProd                                    ptc_rp2;
    reco::EcalIsolatedParticleCandidateRefVector                                  ptc_rv2;
    edm::Wrapper<reco::EcalIsolatedParticleCandidateCollection>                   ptc_wc2;
    edm::reftobase::Holder<reco::Candidate, reco::EcalIsolatedParticleCandidateRef> ptc_h2;

    reco::HcalIsolatedTrackCandidate                                              ptc3;
    reco::HcalIsolatedTrackCandidateCollection                                    ptc_c3;
    reco::HcalIsolatedTrackCandidateRef                                           ptc_r3;
    reco::HcalIsolatedTrackCandidateRefProd                                       ptc_rp3;
    reco::HcalIsolatedTrackCandidateRefVector                                     ptc_rv3;
    edm::Wrapper<reco::HcalIsolatedTrackCandidateCollection>                      ptc_wc3;
    edm::reftobase::Holder<reco::Candidate, reco::HcalIsolatedTrackCandidateRef>  ptc_h3;
    edm::Ref<reco::HcalIsolatedTrackCandidateCollection> 	      	          ptc_sr3;
    reco::HcalIsolatedTrackCandidateSimpleRefVector 	                          ptc_srv3;

  };
}
