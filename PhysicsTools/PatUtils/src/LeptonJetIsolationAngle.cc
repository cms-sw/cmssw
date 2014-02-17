//
// $Id: LeptonJetIsolationAngle.cc,v 1.5 2009/05/26 08:54:22 fabiocos Exp $
//

#include "PhysicsTools/PatUtils/interface/LeptonJetIsolationAngle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <vector>


using namespace pat;


// constructor
LeptonJetIsolationAngle::LeptonJetIsolationAngle() {
}


// destructor
LeptonJetIsolationAngle::~LeptonJetIsolationAngle() {
}


// calculate the JetIsoA for the lepton object
float LeptonJetIsolationAngle::calculate(const Electron & theElectron, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  CLHEP::HepLorentzVector theElectronHLV(theElectron.px(), theElectron.py(), theElectron.pz(), theElectron.energy());
  return this->calculate(theElectronHLV, trackHandle, iEvent);
}
float LeptonJetIsolationAngle::calculate(const Muon & theMuon, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  CLHEP::HepLorentzVector theMuonHLV(theMuon.px(), theMuon.py(), theMuon.pz(), theMuon.energy());
  return this->calculate(theMuonHLV, trackHandle, iEvent);
}


// calculate the JetIsoA for the lepton's HLV
float LeptonJetIsolationAngle::calculate(const CLHEP::HepLorentzVector & aLepton, const edm::Handle<edm::View<reco::Track> > & trackHandle, const edm::Event & iEvent) {
  // FIXME: this is an ugly temporary workaround, JetMET+egamma should come up with a better tool
  // retrieve the jets
  edm::Handle<reco::CaloJetCollection> jetHandle;
  iEvent.getByLabel("iterativeCone5CaloJets", jetHandle);
  reco::CaloJetCollection jetColl = *(jetHandle.product());
  // retrieve the electrons which might be in the jet list
  edm::Handle<std::vector<reco::GsfElectron> > electronsHandle;
  iEvent.getByLabel("pixelMatchGsfElectrons", electronsHandle);
  std::vector<reco::GsfElectron> electrons = *electronsHandle;
  // determine the set of isolated electrons
  std::vector<Electron> isoElectrons;
  for (size_t ie=0; ie<electrons.size(); ie++) {
    Electron anElectron(electrons[ie]);
    if (anElectron.pt() > 10 &&
        trkIsolator_.calculate(anElectron, *trackHandle) < 3.0) {
      isoElectrons.push_back(electrons[ie]);
    }
  }
  // determine the collections of jets, cleaned from electrons
  std::vector<reco::CaloJet> theJets;
  for (reco::CaloJetCollection::const_iterator itJet = jetColl.begin(); itJet != jetColl.end(); itJet++) {
    float mindr2 = 9999.;
    for (size_t ie = 0; ie < isoElectrons.size(); ie++) {
      float dr2 = ::deltaR2(*itJet, isoElectrons[ie]);
      if (dr2 < mindr2) mindr2 = dr2;
    }
    float mindr = std::sqrt(mindr2);
    // yes, all cuts hardcoded buts, but it's a second-order effect
    if (itJet->et() > 15 && mindr > 0.3) theJets.push_back(reco::CaloJet(*itJet));
  }
  // calculate finally the isolation angle
  float isoAngle = 1000; // default to some craze impossible number to inhibit compiler warnings
  for (std::vector<reco::CaloJet>::const_iterator itJet = theJets.begin(); itJet != theJets.end(); itJet++) {
    float curDR = this->spaceAngle(aLepton, *itJet);
    if (curDR < isoAngle) isoAngle = curDR;
  }
  return isoAngle;
}


// calculate the angle between two vectors in 3d eucledian space
float LeptonJetIsolationAngle::spaceAngle(const CLHEP::HepLorentzVector & aLepton, const reco::CaloJet & aJet) {
  return acos(sin(aJet.theta()) * cos(aJet.phi()) * sin(aLepton.theta()) * cos(aLepton.phi())
            + sin(aJet.theta()) * sin(aJet.phi()) * sin(aLepton.theta()) * sin(aLepton.phi())
            + cos(aJet.theta()) * cos(aLepton.theta()));
}
