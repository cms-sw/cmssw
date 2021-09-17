/*HLTTauMCProducer
Producer that creates LorentzVector Collections
from offline reconstructed quantities to be used
in Offline Trigger DQM etc
*/

#ifndef HLTTauMCProducer_h
#define HLTTauMCProducer_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HepMC/GenEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "TLorentzVector.h"
#include <string>
#include <vector>

typedef math::XYZTLorentzVectorD LorentzVector;
typedef std::vector<LorentzVector> LorentzVectorCollection;

class HLTTauMCProducer : public edm::global::EDProducer<> {
public:
  explicit HLTTauMCProducer(const edm::ParameterSet &);

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

private:
  void getGenDecayProducts(const reco::GenParticleRef &,
                           reco::GenParticleRefVector &,
                           int status = 1,
                           int pdgId = 0) const;

  enum tauDecayModes {
    kElectron,
    kMuon,
    kOneProng0pi0,
    kOneProng1pi0,
    kOneProng2pi0,
    kThreeProng0pi0,
    kThreeProng1pi0,
    kOther,
    kUndefined
  };

  const edm::EDGetTokenT<reco::GenParticleCollection> MC_;
  const edm::EDGetTokenT<reco::GenMETCollection> MCMET_;
  const double ptMinMCTau_;
  const double ptMinMCElectron_;
  const double ptMinMCMuon_;
  const std::vector<int> m_PDG_;
  const double etaMin_, etaMax_, phiMin_, phiMax_;
};

#endif
