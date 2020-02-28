#ifndef HZZ4lFilter_H
#define HZZ4lFilter_H

// -*- C++ -*-
//
// Package:    HZZ4lFilter
// Class:      HZZ4lFilter
//
/**\class HZZ4lFilter HZZ4lFilter.cc IOMC/HZZ4lFilter/src/HZZ4lFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Puljak Ivica
//         Created:  Wed Apr 18 12:52:31 CEST 2007
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

//
// class declaration
//
namespace edm {
  class HepMCProduct;
}

class HZZ4lFilter : public edm::EDFilter {
public:
  explicit HZZ4lFilter(const edm::ParameterSet&);
  ~HZZ4lFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  // virtual void endJob() ;

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::HepMCProduct> token_;

  double minPtElectronMuon;
  double maxEtaElectronMuon;
};

#endif
