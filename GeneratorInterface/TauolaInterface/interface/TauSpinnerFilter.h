#ifndef Gen_TauolaInterface_TauSpinnerFilter_H
#define Gen_TauolaInterface_TauSpinnerFilter_H

// I. M. Nugent
// Filter on TauSpinner polarization weights to make unweighted polarized MC 


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "CLHEP/Random/RandomEngine.h"

class TauSpinnerFilter : public edm::EDFilter {
 public:
  TauSpinnerFilter(const edm::ParameterSet&);
  ~TauSpinnerFilter(){};

  virtual bool filter(edm::Event& e, edm::EventSetup const& es);
  void setRandomEngine(CLHEP::HepRandomEngine* v) { fRandomEngine = v; }

 private:
  edm::InputTag src_;
  CLHEP::HepRandomEngine* fRandomEngine;
  double ntaus_;
  edm::EDGetTokenT<double> WTToken_;
};

#endif
