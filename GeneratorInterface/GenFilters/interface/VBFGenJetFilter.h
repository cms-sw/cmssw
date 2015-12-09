#ifndef VBFGenJetFilter_h
#define VBFGenJetFilter_h

// CMSSW include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/JetReco/interface/GenJetCollection.h"

// ROOT includes
#include "TFile.h"
#include "TH1D.h"

// C++ include files
#include <memory>
#include <map>

//
// class declaration
//

class VBFGenJetFilter : public edm::EDFilter {
public:
  explicit VBFGenJetFilter(const edm::ParameterSet&);
  ~VBFGenJetFilter();
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);
private:
  
  // ----------memeber function----------------------
  int    charge      (const int& Id);
  std::vector<HepMC::GenParticle*> getVisibleDecayProducts(HepMC::GenParticle* particle);
  std::vector<HepMC::GenParticle*> getNu (const HepMC::GenEvent* particles);
  std::vector<HepMC::GenParticle*> getSt3(const HepMC::GenEvent* particles);
  void printGenVector(std::vector<HepMC::GenParticle*> vec);
  double nuMET(std::vector<HepMC::GenParticle*> vNu);
  
  std::vector<const reco::GenJet*> filterGenJets(const std::vector<reco::GenJet>* jets);
  //   std::vector<const reco::GenJet*> filterGenJets(const std::vector<reco::GenJet>* jets);
  
  //**************************
  // Private Member data *****
private:
  

  
  // Dijet cut
  bool   oppositeHemisphere;
  double ptMin;
  double etaMin;
  double etaMax;
  double minInvMass;
  double maxInvMass;
  double minDeltaPhi;
  double maxDeltaPhi;
  double minDeltaEta;
  double maxDeltaEta;
  
  // Input tags
  edm::EDGetTokenT< reco::GenJetCollection > m_inputTag_GenJetCollection;
  

};

#endif
