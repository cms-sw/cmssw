#ifndef PythiaHLTSoupFilter_h
#define PythiaHLTSoupFilter_h
// -*- C++ -*-
//
// Package:    PythiaHLTSoupFilter
// Class:      PythiaHLTSoupFilter
//
/**\class PythiaHLTSoupFilter PythiaHLTSoupFilter.cc IOMC/PythiaHLTSoupFilter/src/PythiaHLTSoupFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Jan 23 14:57:54 CET 2006
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
// class decleration
//
namespace edm {
  class HepMCProduct;
}

class PythiaHLTSoupFilter : public edm::EDFilter {
public:
  explicit PythiaHLTSoupFilter(const edm::ParameterSet&);
  ~PythiaHLTSoupFilter() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::HepMCProduct> token_;

  double minptelectron;
  double minptmuon;
  double maxetaelectron;
  double maxetamuon;
  double minpttau;
  double maxetatau;
};
#endif
