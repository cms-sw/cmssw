#ifndef PYTHIAFILTERHT_h
#define PYTHIAFILTERHT_h
// -*- C++ -*-
//
// Package:    PythiaFilterHT
// Class:      PythiaFilterHT
//
/**\class PythiaFilterHT PythiaFilterHT.cc IOMC/PythiaFilterHT/src/PythiaFilterHT.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alejandro Gomez Espinosa
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

namespace edm {
  class HepMCProduct;
}
//
// class decleration
//

class PythiaFilterHT : public edm::EDFilter {
public:
  explicit PythiaFilterHT(const edm::ParameterSet&);
  ~PythiaFilterHT() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::HepMCProduct> label_;
  int particleID;
  double minpcut;
  double maxpcut;
  double minptcut;
  double minhtcut;
  double maxptcut;
  double minetacut;
  double maxetacut;
  double minrapcut;
  double maxrapcut;
  double maxphicut;
  double minphicut;

  double rapidity;

  int status;
  int motherID;
  int processID;
  int theNumberOfTestedEvt;
  int theNumberOfSelected;
  int maxnumberofeventsinrun;
};
#endif
