#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

EDM_REGISTER_PLUGINFACTORY(CutApplicatorFactory,"CutApplicatorFactory");

CutApplicatorBase::result_type 
CutApplicatorBase::
operator()(const CutApplicatorBase::argument_type& arg) const {  
  switch(candidateType()) {
  case ELECTRON:
    {      
      const reco::GsfElectronRef ele(arg.id(),arg.key(),arg.productGetter());
      return this->operator()(ele);
    }
    break;
  case MUON:
    {
      const reco::MuonRef mu(arg.id(),arg.key(),arg.productGetter());
      return this->operator()(mu);
    }
    break;
  case PHOTON:
    {
      const reco::PhotonRef pho(arg.id(),arg.key(),arg.productGetter());
      return this->operator()(pho);
    }
    break;
  case TAU:
    {
      const reco::PFTauRef tau(arg.id(),arg.key(),arg.productGetter());
      return this->operator()(tau);
    }
    break;
  case PATELECTRON:
    {
      const pat::ElectronRef ele(arg.id(),arg.key(),arg.productGetter());
      return this->operator()(ele);
    }
    break;
  case PATMUON:
    {
      const pat::MuonRef mu(arg.id(),arg.key(),arg.productGetter());
      return this->operator()(mu);
    }
    break;
  case PATPHOTON:
    {
      const pat::PhotonRef pho(arg.id(),arg.key(),arg.productGetter());
      return this->operator()(pho);
    }
    break;
  case PATTAU:
    {
      const pat::TauRef tau(arg.id(),arg.key(),arg.productGetter());
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
