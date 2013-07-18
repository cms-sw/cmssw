//
// $Id: PATObjectSelector.h,v 1.7 2010/06/17 10:08:39 gpetrucc Exp $
//

#ifndef PhysicsTools_PatAlgos_PATObjectSelector_h
#define PhysicsTools_PatAlgos_PATObjectSelector_h

#include "DataFormats/Common/interface/RefVector.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PFParticle.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

#include "PhysicsTools/PatAlgos/plugins/PATJetSelector.h"

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
  /* typedef SingleObjectSelector< */
  /*             std::vector<Jet>, */
  /*             StringCutObjectSelector<Jet> */
  /*         > PATJetSelector; */
  typedef SingleObjectSelector<
              std::vector<MET>,
              StringCutObjectSelector<MET>
          > PATMETSelector;
  typedef SingleObjectSelector<
              std::vector<PFParticle>,
              StringCutObjectSelector<PFParticle>
          > PATPFParticleSelector;
  typedef SingleObjectSelector<
              std::vector<CompositeCandidate>,
              StringCutObjectSelector<CompositeCandidate, true> // true => lazy parsing => get all methods of daughters
          > PATCompositeCandidateSelector;
  typedef SingleObjectSelector<
              std::vector<TriggerObjectStandAlone>,
              StringCutObjectSelector<TriggerObjectStandAlone>
          > PATTriggerObjectStandAloneSelector;
  typedef SingleObjectSelector<
              std::vector<GenericParticle>,
              StringCutObjectSelector<GenericParticle>
          > PATGenericParticleSelector;

  typedef SingleObjectSelector<
              std::vector<Electron>,
              StringCutObjectSelector<Electron>,
              edm::RefVector<std::vector<Electron> >
          > PATElectronRefSelector;
  typedef SingleObjectSelector<
              std::vector<Muon>,
              StringCutObjectSelector<Muon>,
              edm::RefVector<std::vector<Muon> >
          > PATMuonRefSelector;
  typedef SingleObjectSelector<
              std::vector<Tau>,
              StringCutObjectSelector<Tau>,
              edm::RefVector<std::vector<Tau> >
          > PATTauRefSelector;
  typedef SingleObjectSelector<
              std::vector<Photon>,
              StringCutObjectSelector<Photon>,
              edm::RefVector<std::vector<Photon> >
          > PATPhotonRefSelector;
  typedef SingleObjectSelector<
              std::vector<Jet>,
              StringCutObjectSelector<Jet>,
              edm::RefVector<std::vector<Jet> >
          > PATJetRefSelector;
  typedef SingleObjectSelector<
              std::vector<MET>,
              StringCutObjectSelector<MET>,
              edm::RefVector<std::vector<MET> >
          > PATMETRefSelector;
  typedef SingleObjectSelector<
              std::vector<PFParticle>,
              StringCutObjectSelector<PFParticle>,
              edm::RefVector<std::vector<PFParticle> >
          > PATPFParticleRefSelector;
  typedef SingleObjectSelector<
              std::vector<GenericParticle>,
              StringCutObjectSelector<GenericParticle>,
              edm::RefVector<std::vector<GenericParticle> >
          > PATGenericParticleRefSelector;
  typedef SingleObjectSelector<
              std::vector<CompositeCandidate>,
              StringCutObjectSelector<CompositeCandidate, true>, // true => lazy parsing => get all methods of daughters
              edm::RefVector<std::vector<CompositeCandidate> >
          > PATCompositeCandidateRefSelector;



}

#endif
