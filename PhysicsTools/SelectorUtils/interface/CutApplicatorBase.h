#ifndef PhysicsTools_SelectorUtils_CutApplicatorBase_h
#define PhysicsTools_SelectorUtils_CutApplicatorBase_h

//
//
//

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Common/interface/EventBase.h"

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

namespace reco {
  typedef edm::Ptr<reco::GsfElectron> GsfElectronPtr;
  typedef edm::Ptr<reco::Photon> PhotonPtr;
  typedef edm::Ptr<reco::Muon> MuonPtr;
  typedef edm::Ptr<reco::PFTau> PFTauPtr;
}

namespace pat {
  typedef edm::Ptr<pat::Electron> ElectronPtr;
  typedef edm::Ptr<pat::Photon> PhotonPtr;
  typedef edm::Ptr<pat::Muon> MuonPtr;
  typedef edm::Ptr<pat::Tau> TauPtr;
}

class CutApplicatorBase : public candf::CandidateCut {
 public:
  enum CandidateType{NONE,
		     ELECTRON,MUON,PHOTON,TAU,
		     PATELECTRON,PATMUON,PATPHOTON,PATTAU};

 CutApplicatorBase(): CandidateCut() {}

 CutApplicatorBase(const edm::ParameterSet& c) :
  _name(c.getParameter<std::string>("cutName")) {
  }
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  CutApplicatorBase(const CutApplicatorBase&) = delete;
  CutApplicatorBase& operator=(const CutApplicatorBase&) = delete;
#endif
    
  
  virtual result_type operator()(const argument_type&) const 
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    final
#endif
    ;
  
  // electrons 
  virtual result_type operator()(const reco::GsfElectronPtr&) const {return false;}
  virtual result_type operator()(const pat::ElectronPtr&) const {return false;}

  // photons
  virtual result_type operator()(const reco::PhotonPtr&) const {return false;}
  virtual result_type operator()(const pat::PhotonPtr&) const {return false;}
  
  // muons
  virtual result_type operator()(const reco::MuonPtr&) const {return false;}
  virtual result_type operator()(const pat::MuonPtr&) const {return false;}

  // taus
  virtual result_type operator()(const reco::PFTauPtr&) const {return false;}
  virtual result_type operator()(const pat::TauPtr&) const {return false;}

  // candidate operation
  virtual result_type asCandidate(const argument_type&) const {return false;} 
  
  virtual CandidateType candidateType() const { return NONE; }

  virtual const std::string& name() const { return _name; }
  
  //! Destructor
  virtual ~CutApplicatorBase(){};
  
 private:
  const std::string _name;
  
};

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< CutApplicatorBase* (const edm::ParameterSet&) > CutApplicatorFactory;
#endif

#endif
