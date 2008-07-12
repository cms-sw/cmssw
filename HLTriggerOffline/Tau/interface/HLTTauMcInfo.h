#ifndef HLTTauMcInfo_h
#define HLTTauMcInfo_h

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

class HLTTauMcInfo : public edm::EDProducer {

public:
  explicit HLTTauMcInfo(const edm::ParameterSet&);
  ~HLTTauMcInfo();

  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  typedef math::XYZTLorentzVectorD LorentzVector;
  typedef std::vector<LorentzVector> LorentzVectorCollection;
  edm::InputTag genParticles;
  double etaMax;
  double ptMin;
  int m_PDG;
  edm::InputTag pfTauCollection_, pfTauDiscriminatorProd_;
  bool usePFTauMatching_;

};

#endif
