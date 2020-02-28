// -*- C++ -*-
//
// Package:    TwoVBGenGenFilter
// Class:      TwoVBGenGenFilter
//
/**\class TwoVBGenGenFilter TwoVBGenGenFilter.cc GeneratorInterface/GenFilters/src/TwoVBGenGenFilter.cc

 Description: select semileptonic double-VB events

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Maurizio Pierini, Thiago Tomei
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//
// class declaration
//

class TwoVBGenFilter : public edm::EDFilter {
public:
  explicit TwoVBGenFilter(const edm::ParameterSet&);
  ~TwoVBGenFilter() override;

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::InputTag src_;
  bool eejj_, enujj_, nunujj_, mumujj_, munujj_, tautaujj_, taunujj_;

  bool isNeutrino(const HepMC::GenParticle*);
  bool isQuark(const HepMC::GenParticle*);
  bool isElectron(const HepMC::GenParticle*);
  bool isMuon(const HepMC::GenParticle*);
  bool isTau(const HepMC::GenParticle*);
};
