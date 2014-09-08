#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"

EDM_REGISTER_PLUGINFACTORY(CutApplicatorFactory,"CutApplicatorFactory");

CutApplicatorBase::result_type 
CutApplicatorBase::
operator()(const CutApplicatorBase::argument_type& arg) const {
  if( arg.isNull() ) { 
    throw cms::Exception("BadProductPtr")
      << _name << "received a bad product ref to process!" << std::endl;
  }
  
  switch(candidateType()) {
  case ELECTRON:
    {      
      const reco::GsfElectronPtr ele(arg);
      return this->operator()(ele);
    }
    break;
  case MUON:
    {
      const reco::MuonPtr mu(arg);
      return this->operator()(mu);
    }
    break;
  case PHOTON:
    {
      const reco::PhotonPtr pho(arg);
      return this->operator()(pho);
    }
    break;
  case TAU:
    {
      const reco::PFTauPtr tau(arg);
      return this->operator()(tau);
    }
    break;
  case PATELECTRON:
    {
      const pat::ElectronPtr ele(arg);
      return this->operator()(ele);
    }
    break;
  case PATMUON:
    {
      const pat::MuonPtr mu(arg);
      return this->operator()(mu);
    }
    break;
  case PATPHOTON:
    {
      const pat::PhotonPtr pho(arg);
      return this->operator()(pho);
    }
    break;
  case PATTAU:
    {
      const pat::TauPtr tau(arg);
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
