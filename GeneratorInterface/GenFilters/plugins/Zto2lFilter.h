#ifndef Zto2lFilter_H
#define Zto2lFilter_H

// -*- C++ -*-
//
// Package:    Zto2lFilter
// Class:      Zto2lFilter
//
/**\class Zto2lFilter Zto2lFilter.cc Zbb/Zto2lFilter/src/Zto2lFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Aruna Nayak
//         Created:  Thu Aug 23 11:37:45 CEST 2007
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

class Zto2lFilter : public edm::EDFilter {
public:
  explicit Zto2lFilter(const edm::ParameterSet&);
  ~Zto2lFilter() override;

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  std::string fLabel_;
  double minInvariantMass_, maxEtaLepton_;
};
#endif
