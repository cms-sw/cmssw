//
// $Id: PATObjectSelector.h,v 1.2 2008/02/07 18:20:08 lowette Exp $
//

#ifndef PhysicsTools_PatAlgos_PATObjectSelector_h
#define PhysicsTools_PatAlgos_PATObjectSelector_h


#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Particle.h"

#include <vector>


namespace pat {


  typedef SingleObjectSelector<
              std::vector<Electron>,
              StringCutObjectSelector<Electron>
          > PATElectronSelector;
  typedef SingleObjectSelector<
              std::vector<Muon>,
              StringCutObjectSelector<Muon>
          > PATMuonSelector;
  typedef SingleObjectSelector<
              std::vector<Tau>,
              StringCutObjectSelector<Tau>
          > PATTauSelector;
  typedef SingleObjectSelector<
              std::vector<Photon>,
              StringCutObjectSelector<Photon>
          > PATPhotonSelector;
  typedef SingleObjectSelector<
              std::vector<Jet>,
              StringCutObjectSelector<Jet>
          > PATJetSelector;
  typedef SingleObjectSelector<
              std::vector<MET>,
              StringCutObjectSelector<MET>
          > PATMETSelector;
  typedef SingleObjectSelector<
              std::vector<Particle>,
              StringCutObjectSelector<Particle>
          > PATParticleSelector;


}

#endif
