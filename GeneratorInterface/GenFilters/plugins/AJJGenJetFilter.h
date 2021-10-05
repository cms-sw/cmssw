#ifndef AJJGenJetAnalyzer_h
#define AJJGenJetAnalyzer_h

// CMSSW include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

// ROOT includes
#include "TFile.h"
#include "TH1D.h"
#include "TH2D.h"

// C++ include files
#include <memory>
#include <map>
#include <vector>

using namespace edm;
using namespace std;
//
// class declaration
//

class AJJGenJetFilter : public edm::EDFilter {
public:
  explicit AJJGenJetFilter(const edm::ParameterSet& pset);
  ~AJJGenJetFilter() override;

  //void analyze(edm::Event&, const edm::EventSetup&) 
  bool filter(edm::Event&, const edm::EventSetup&) override;
private:
  // ----------memeber function----------------------
  std::vector<const reco::GenJet*> filterGenJets(const vector<reco::GenJet>* jets);
  std::vector<const reco::GenParticle*> filterGenLeptons(const std::vector<reco::GenParticle>* particles);
  std::vector<const reco::GenParticle*> filterGenPhotons(const std::vector<reco::GenParticle>* particles);

  //**************************
  // Private Member data *****
private:
  // Dijet cut
  double ptMin;
  double etaMin;
  double etaMax;
  double minDeltaEta;
  double maxDeltaEta;
  double deltaRJetLep;
  double maxPhotonEta;
  double minPhotonPt;
  double maxPhotonPt;
  double mininvmass;

  // Input tags
  edm::EDGetTokenT<reco::GenJetCollection> m_GenJetCollection;
  edm::EDGetTokenT<reco::GenParticleCollection> m_GenParticleCollection;
};

#endif
