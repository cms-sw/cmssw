/*HLTTauMCProducer
Producer that creates LorentzVector Collections
from offline reconstructed quantities to be used
in Offline Trigger DQM etc
*/

#ifndef HLTTauMCProducer_h
#define HLTTauMCProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "HepMC/GenEvent.h"
#include <vector>
#include <string>

typedef math::XYZTLorentzVectorD LorentzVector;
typedef std::vector<LorentzVector> LorentzVectorCollection;

class HLTTauMCProducer : public edm::EDProducer {
  
public:
  explicit HLTTauMCProducer(const edm::ParameterSet&);
  ~HLTTauMCProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:

  std::vector<HepMC::GenParticle*> getGenStableDecayProducts(const HepMC::GenParticle* particle);

  enum tauDecayModes {kElectron, kMuon, 
		      kOneProng0pi0, kOneProng1pi0, kOneProng2pi0,
		      kThreeProng0pi0, kThreeProng1pi0,
		      kOther, kUndefined};

  edm::InputTag MC_;
  double ptMinMCTau_;
  double ptMinMCElectron_;
  double ptMinMCMuon_;
  std::vector<int> m_PDG_;
  double etaMax;

};

#endif
