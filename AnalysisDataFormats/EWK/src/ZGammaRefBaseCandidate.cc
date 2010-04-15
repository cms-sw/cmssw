#include "AnalysisDataFormats/EWK/interface/ZGammaRefBaseCandidate.h"

#include "FWCore/Utilities/interface/EDMException.h"

namespace ewk
{

  ZGammaRefBaseCandidate::ZGammaRefBaseCandidate(const reco::CandidateBaseRef& theZ,
						 const reco::CandidateBaseRef& photon)
  {
    addDaughter(theZ);
    addDaughter(photon);
  }
  
  ZGammaRefBaseCandidate::ZGammaRefBaseCandidate(const edm::Ref<ewk::DiLeptonRefBaseCandidateCollection>& theZ,
						 const edm::Ref<reco::PhotonCollection>& photon)
    
  {
    addDaughter(reco::CandidateBaseRef(theZ));
    addDaughter(reco::CandidateBaseRef(photon));
  }
  
  const edm::Ref<ewk::DiLeptonRefBaseCandidateCollection> ZGammaRefBaseCandidate::zDaughterptr() const
  {
    edm::Ref<ewk::DiLeptonRefBaseCandidateCollection> d;
    for(unsigned i = 0; i < numberOfDaughters(); ++i)
      {
	try
	  {
	    d = daughterRef(i).castTo<edm::Ref<ewk::DiLeptonRefBaseCandidateCollection> >();
	  }
	catch(edm::Exception& e)
	  {
	    continue;
	  }
      }
    return d;
  }

  const ewk::DiLeptonRefBaseCandidate ZGammaRefBaseCandidate::zDaughter() const
  {
    edm::Ref<ewk::DiLeptonRefBaseCandidateCollection> d = zDaughterptr();
    if(d.isNonnull()) return *d;
    return ewk::DiLeptonRefBaseCandidate();
  }
    
  const edm::Ref<reco::PhotonCollection> ZGammaRefBaseCandidate::photonDaughterptr() const
  {
    edm::Ref<reco::PhotonCollection> d;
    for(unsigned i = 0; i < numberOfDaughters(); ++i)
      {
	try
	  {
	    d = daughterRef(i).castTo<edm::Ref<reco::PhotonCollection> >();
	  }
	catch(edm::Exception& e)
	  {
	    continue;
	  }
      }
    return d;
  }
  
  const reco::Photon ZGammaRefBaseCandidate::photonDaughter() const
  {
    edm::Ref<reco::PhotonCollection> d = photonDaughterptr();    
    if(d.isNonnull()) return *d;
    return reco::Photon();
  }
}

