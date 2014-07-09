#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

EDM_REGISTER_PLUGINFACTORY(CutApplicatorFactory,"CutApplicatorFactory");

CutApplicatorBase::result_type 
CutApplicatorBase::
operator()(const CutApplicatorBase::argument_type& arg) const {  
  switch(candidateType()) {
  case ELECTRON:
    {
      const reco::GsfElectron& ele = static_cast<const reco::GsfElectron&>(arg);
      return this->operator()(ele);
    }
    break;
  case MUON:
    {
      const reco::Muon& mu = static_cast<const reco::Muon&>(arg);
      return this->operator()(mu);
    }
    break;
  case PHOTON:
    {
      const reco::Photon& pho = static_cast<const reco::Photon&>(arg);
      return this->operator()(pho);
    }
    break;
  case TAU:
    {
      const reco::PFTau& tau = static_cast<const reco::PFTau&>(arg);
      return this->operator()(tau);
    }
    break;
  case PATELECTRON:
    {
      const pat::Electron& ele = static_cast<const pat::Electron&>(arg);
      return this->operator()(ele);
    }
    break;
  case PATMUON:
    {
      const pat::Muon& mu = static_cast<const pat::Muon&>(arg);
      return this->operator()(mu);
    }
    break;
  case PATPHOTON:
    {
      const pat::Photon& pho = static_cast<const pat::Photon&>(arg);
      return this->operator()(pho);
    }
    break;
  case PATTAU:
    {
      const pat::Tau& tau = static_cast<const pat::Tau&>(arg);
      return this->operator()(tau);
    }
    break;
  case NONE:
    return asCandidate(arg);
    break;
  default:
    throw cms::Exception("BadCandidateType")
      << "Unknown candidate type";
  }
}
