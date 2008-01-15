//
// $Id: LeptonJetIsolationAngle.h,v 1.1 2008/01/07 11:48:26 lowette Exp $
//

#ifndef PhysicsTools_PatUtils_LeptonJetIsolationAngle_h
#define PhysicsTools_PatUtils_LeptonJetIsolationAngle_h

/**
  \class    LeptonJetIsolationAngle LeptonJetIsolationAngle.h "PhysicsTools/PatUtils/interface/LeptonJetIsolationAngle.h"
  \brief    Calculates a lepton's jet isolation angle

   LeptonJetIsolationAngle calculates an isolation angle w.r.t. a list of
   given jets as the minimal angle to a jet in Euclidean space, as defined in
   CMS Note 2006/024

  \author   Steven Lowette
  \version  $Id: LeptonJetIsolationAngle.h,v 1.1 2008/01/07 11:48:26 lowette Exp $
*/


#include "DataFormats/FWLite/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "CLHEP/Vector/LorentzVector.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "PhysicsTools/PatUtils/interface/TrackerIsolationPt.h"


namespace pat {


  class LeptonJetIsolationAngle {

    public:

      LeptonJetIsolationAngle();
      ~LeptonJetIsolationAngle();

      float calculate(const Electron & anElectron, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      float calculate(const Muon & aMuon, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);

    private:

      float calculate(const HepLorentzVector & aLepton, const edm::Handle<reco::TrackCollection> & trackHandle, const edm::Event & iEvent);
      float spaceAngle(const HepLorentzVector & aLepton, const reco::CaloJet & aJet);

    private:

      TrackerIsolationPt trkIsolator_;

  };


}

#endif

