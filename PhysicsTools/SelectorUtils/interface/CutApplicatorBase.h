#ifndef PhysicsTools_SelectorUtils_CutApplicatorBase_h
#define PhysicsTools_SelectorUtils_CutApplicatorBase_h

//
//
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"

#include "PhysicsTools/SelectorUtils/interface/CandidateCut.h"

namespace candf = candidate_functions;

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TauReco/interface/PFTau.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

class CutApplicatorBase : public candf::CandidateCut {
 public:
  enum CandidateType{NONE,
		     ELECTRON,MUON,PHOTON,TAU,
		     PATELECTRON,PATMUON,PATPHOTON,PATTAU};
  
 CutApplicatorBase(const edm::ParameterSet& c) :
  _name(c.getParameter<std::string>("cutName")) {
  }
  
  CutApplicatorBase(const CutApplicatorBase&) = delete;
  CutApplicatorBase& operator=(const CutApplicatorBase&) = delete;
    
  virtual result_type operator()(const argument_type&) const final;
  
  // electrons 
  virtual result_type operator()(const reco::GsfElectronRef&) const {return false;}
  virtual result_type operator()(const pat::ElectronRef&) const {return false;}

  // photons
  virtual result_type operator()(const reco::PhotonRef&) const {return false;}
  virtual result_type operator()(const pat::PhotonRef&) const {return false;}
  
  // muons
  virtual result_type operator()(const reco::MuonRef&) const {return false;}
  virtual result_type operator()(const pat::MuonRef&) const {return false;}

  // taus
  virtual result_type operator()(const reco::PFTauRef&) const {return false;}
  virtual result_type operator()(const pat::TauRef&) const {return false;}

  // candidate operation
  virtual result_type asCandidate(const argument_type&) const {return false;} 
  
  virtual CandidateType candidateType() const { return NONE; }

  const std::string& name() const { return _name; }
  
  //! Destructor
  virtual ~CutApplicatorBase(){};
  
 private:
  const std::string _name;
  
};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< CutApplicatorBase* (const edm::ParameterSet&) > CutApplicatorFactory;

#endif
