#ifndef ZgammaMassFilter_h
#define ZgammaMassFilter_h
// -*- C++ -*-
//
// Package:    ZgammaMassFilter
// Class:      ZgammaMassFilter
//
/* 

 Description: filter events based on the Pythia particle information

 Implementation: inherits from generic EDFilter
     
*/
//
// Original Author:  Alexey Ferapontov
//         Created:  Thu July 26 11:57:54 CDT 2012
// $Id: ZgammaMassFilter.h,v 1.1 2012/08/10 12:46:29 lenzip Exp $
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

class ZgammaMassFilter : public edm::EDFilter {
public:
  explicit ZgammaMassFilter(const edm::ParameterSet&);
  ~ZgammaMassFilter() override;

  bool filter(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------memeber function----------------------
  int charge(const int& Id);

  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::HepMCProduct> token_;

  double minPhotonPt;
  double minLeptonPt;

  double minPhotonEta;
  double minLeptonEta;

  double maxPhotonEta;
  double maxLeptonEta;

  double minDileptonMass;
  double minZgMass;
};
#endif
