#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

EDM_REGISTER_PLUGINFACTORY(CutApplicatorFactory,"CutApplicatorFactory");

CutApplicatorBase::result_type 
CutApplicatorBase::
operator()(const CutApplicatorBase::argument_type& arg) const {
  if( arg.isNull() ) { 
    throw cms::Exception("BadProductRef")
      << _name << "received a bad product ref to process!" << std::endl;
  }
  
  switch(candidateType()) {
  case ELECTRON:
    {      
      const reco::GsfElectronRef ele = arg.castTo<reco::GsfElectronRef>();
      return this->operator()(ele);
    }
    break;
  case MUON:
    {
      const reco::MuonRef mu = arg.castTo<reco::MuonRef>();
      return this->operator()(mu);
    }
    break;
  case PHOTON:
    {
      const reco::PhotonRef pho = arg.castTo<reco::PhotonRef>();
      return this->operator()(pho);
    }
    break;
  case TAU:
    {
      const reco::PFTauRef tau = arg.castTo<reco::PFTauRef>();
      return this->operator()(tau);
    }
    break;
  case PATELECTRON:
    {
      const pat::ElectronRef ele = arg.castTo<pat::ElectronRef>();
      return this->operator()(ele);
    }
    break;
  case PATMUON:
    {
      const pat::MuonRef mu = arg.castTo<pat::MuonRef>();
      return this->operator()(mu);
    }
    break;
  case PATPHOTON:
    {
      const pat::PhotonRef pho = arg.castTo<pat::PhotonRef>();
      return this->operator()(pho);
    }
    break;
  case PATTAU:
    {
      const pat::TauRef tau = arg.castTo<pat::TauRef>();
      return this->operator()(tau);
    }
    break;
  case NONE:
    {
      return asCandidate(arg);
      break;
    }
  default:
    throw cms::Exception("BadCandidateType")
      << "Unknown candidate type";
  }
}
