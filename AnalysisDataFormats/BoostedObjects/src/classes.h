#include "AnalysisDataFormats/BoostedObjects/interface/GenParticle.h" 
#include "AnalysisDataFormats/BoostedObjects/interface/GenParticleWithDaughters.h" 
#include "AnalysisDataFormats/BoostedObjects/interface/Jet.h"
#include "AnalysisDataFormats/BoostedObjects/interface/ResolvedVjj.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace AnalysisDataFormats_BoostedObjects {
  struct dictionary {
    vlq::Candidate vlqCandidate ;
    edm::Wrapper<vlq::Candidate> w_vlqCandidate ;
    std::vector<vlq::Candidate> v_vlqCandidate ;
    edm::Wrapper<std::vector<vlq::Candidate> > w_v_vlqCandidate ;

    vlq::CandidateCollection vlqCandidateCollection ;
    edm::Wrapper<vlq::CandidateCollection> w_vlqCandidateCollection ;

    vlq::GenParticle vlqGenParticle ; 
    edm::Wrapper<vlq::GenParticle> w_vlqGenParticle ;
    std::vector<vlq::GenParticle> v_vlqGenParticle ;
    edm::Wrapper<std::vector<vlq::GenParticle> > w_v_vlqGenParticle ;

    vlq::GenParticleCollection vlqGenParticleCollection ;
    edm::Wrapper<vlq::GenParticleCollection> w_vlqGenParticleCollection ;

    vlq::GenParticleWithDaughters vlqGenParticleWithDaughters ; 
    edm::Wrapper<vlq::GenParticleWithDaughters> w_vlqGenParticleWithDaughters ;
    std::vector<vlq::GenParticleWithDaughters> v_vlqGenParticleWithDaughters ;
    edm::Wrapper<std::vector<vlq::GenParticleWithDaughters> > w_v_vlqGenParticleWithDaughters ;

    vlq::GenParticleWithDaughtersCollection vlqGenParticleWithDaughtersCollection ; 
    edm::Wrapper<vlq::GenParticleWithDaughtersCollection> w_vlqGenParticleWithDaughtersCollection ;

    vlq::Jet vlqJet ;
    edm::Wrapper<vlq::Jet> w_vlqJet ;
    std::vector<vlq::Jet> v_vlqJet ;
    edm::Wrapper<std::vector<vlq::Jet> > w_v_vlqJet ;

    vlq::JetCollection vlqJetCollection ;
    edm::Wrapper<vlq::JetCollection> w_vlqJetCollection ;

    vlq::ResolvedVjj vlqResolvedVjj ;
    edm::Wrapper<vlq::ResolvedVjj> w_vlqResolvedVjj ; 
    std::vector<vlq::ResolvedVjj> v_vlqResolvedVjj ;
    edm::Wrapper<std::vector<vlq::ResolvedVjj> > w_v_vlqResolvedVjj ;

    vlq::ResolvedVjjCollection vlqResolvedVjjCollection ;
    edm::Wrapper<vlq::ResolvedVjjCollection> w_vlqResolvedVjjCollection ; 
  };
}
