#ifndef AnalysisDataFormats_EWK_ZGammaRefBaseCandidate_h
#define AnalysisDataFormats_EWK_ZGammaRefBaseCandidate_h

#include <map>
#include <memory>

#include "AnalysisDataFormats/EWK/interface/DiLeptonRefBaseCandidate.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/Candidate/interface/CompositeRefBaseCandidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"

namespace ewk{
  
  class ZGammaRefBaseCandidate : public reco::CompositeRefBaseCandidate
  {
  public:    
    ZGammaRefBaseCandidate() {};
    ZGammaRefBaseCandidate(const reco::CandidateBaseRef&, const reco::CandidateBaseRef&);
    ZGammaRefBaseCandidate(const edm::Ref<ewk::DiLeptonRefBaseCandidateCollection>&,
			   const edm::Ref<reco::PhotonCollection>&);
    
    virtual ~ZGammaRefBaseCandidate() {};
    
    const edm::Ref<ewk::DiLeptonRefBaseCandidateCollection> zDaughterptr() const;    
    const ewk::DiLeptonRefBaseCandidate zDaughter() const;
    
    const edm::Ref<reco::PhotonCollection> photonDaughterptr() const;    
    const reco::Photon photonDaughter() const;
  };

  typedef std::vector<ZGammaRefBaseCandidate> ZGammaRefBaseCandidateCollection;

}
#endif
