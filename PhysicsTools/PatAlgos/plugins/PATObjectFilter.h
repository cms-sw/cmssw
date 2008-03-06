//
// $Id: PATObjectFilter.h,v 1.2 2008/02/07 18:20:08 lowette Exp $
//

#ifndef PhysicsTools_PatAlgos_PATObjectFilter_h
#define PhysicsTools_PatAlgos_PATObjectFilter_h


#include "PhysicsTools/UtilAlgos/interface/AnySelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/UtilAlgos/interface/MinNumberSelector.h"
#include "PhysicsTools/PatUtils/interface/MaxNumberSelector.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Particle.h"

#include <vector>


namespace pat {

  typedef ObjectCountFilter<std::vector<Electron>, AnySelector, MinNumberSelector> PATElectronMinFilter;
  typedef ObjectCountFilter<std::vector<Muon>,     AnySelector, MinNumberSelector> PATMuonMinFilter;
  typedef ObjectCountFilter<std::vector<Tau>,      AnySelector, MinNumberSelector> PATTauMinFilter;
  typedef ObjectCountFilter<std::vector<Photon>,   AnySelector, MinNumberSelector> PATPhotonMinFilter;
  typedef ObjectCountFilter<std::vector<Jet>,      AnySelector, MinNumberSelector> PATJetMinFilter;
  typedef ObjectCountFilter<std::vector<MET>,      AnySelector, MinNumberSelector> PATMETMinFilter;
  typedef ObjectCountFilter<std::vector<Particle>, AnySelector, MinNumberSelector> PATParticleMinFilter;

  typedef ObjectCountFilter<std::vector<Electron>, AnySelector, MaxNumberSelector> PATElectronMaxFilter;
  typedef ObjectCountFilter<std::vector<Muon>,     AnySelector, MaxNumberSelector> PATMuonMaxFilter;
  typedef ObjectCountFilter<std::vector<Tau>,      AnySelector, MaxNumberSelector> PATTauMaxFilter;
  typedef ObjectCountFilter<std::vector<Photon>,   AnySelector, MaxNumberSelector> PATPhotonMaxFilter;
  typedef ObjectCountFilter<std::vector<Jet>,      AnySelector, MaxNumberSelector> PATJetMaxFilter;
  typedef ObjectCountFilter<std::vector<MET>,      AnySelector, MaxNumberSelector> PATMETMaxFilter;
  typedef ObjectCountFilter<std::vector<Particle>, AnySelector, MaxNumberSelector> PATParticleMaxFilter;

}


#endif
