//
// $Id: CaloIsolationEnergy.cc,v 1.4 2011/11/01 23:35:45 gowdy Exp $
//

#include "PhysicsTools/PatUtils/interface/CaloIsolationEnergy.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include <vector>

using namespace pat;

/// constructor
CaloIsolationEnergy::CaloIsolationEnergy() {
}

/// destructor
CaloIsolationEnergy::~CaloIsolationEnergy() {
}

/// calculate the CalIsoE from the lepton object
float CaloIsolationEnergy::calculate(const Electron & theElectron, const std::vector<CaloTower> & theTowers, float isoConeElectron) const {
  float isoE = this->calculate(*theElectron.gsfTrack(), theElectron.energy(), theTowers, isoConeElectron);
  return isoE - theElectron.caloEnergy();
}
float CaloIsolationEnergy::calculate(const Muon & theMuon, const std::vector<CaloTower> & theTowers, float isoConeMuon) const {
  return this->calculate(*theMuon.track(), theMuon.energy(), theTowers, isoConeMuon);
}


/// calculate the CalIsoE from the lepton's track
float CaloIsolationEnergy::calculate(const reco::Track & theTrack, const float leptonEnergy, const std::vector<CaloTower> & theTowers, float isoCone) const {
  float isoELepton = 0;
  // calculate iso energy
  //const CaloTower * closestTower = 0;
  float closestDR = 10000;
  for (std::vector<CaloTower>::const_iterator itTower = theTowers.begin(); itTower != theTowers.end(); itTower++) {
    // calculate dPhi with correct sign
    float dPhi = theTrack.phi() - itTower->phi();
    if (dPhi > M_PI)  dPhi = -2*M_PI + dPhi;
    if (dPhi < -M_PI) dPhi =  2*M_PI + dPhi;
    // calculate dR
    float dR = sqrt(std::pow(theTrack.eta()-itTower->eta(), 2) + std::pow(dPhi, 2));
    // calculate energy in cone around direction at vertex of the track
    if (dR < isoCone) {
      isoELepton += itTower->energy();
      if (dR < closestDR) {
        closestDR = dR;
        //closestTower = &(*itTower);
      }
    }
  }
  // subtract track deposits from total energy in cone
//  if (closestTower) isoELepton -= closestTower->energy();
  // return the iso energy
  return isoELepton;
}
