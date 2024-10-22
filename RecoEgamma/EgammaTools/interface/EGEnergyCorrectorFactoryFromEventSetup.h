#ifndef RecoEgamma_EgammaTools_EGEnergyCorrectorFactoryFromEventSetup_h
#define RecoEgamma_EgammaTools_EGEnergyCorrectorFactoryFromEventSetup_h
// -*- C++ -*-
//
// Package:     RecoEgamma/EgammaTools
// Class  :     EGEnergyCorrectorFactoryFromEventSetup
//
/**\class EGEnergyCorrectorFactoryFromEventSetup EGEnergyCorrectorFactoryFromEventSetup.h "RecoEgamma/EgammaTools/interface/EGEnergyCorrectorFactoryFromEventSetup.h"

 Description: Initializes a EGEnergyCorrector from data in the DB

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 03 Sep 2021 18:54:38 GMT
//

// system include files

// user include files
#include "EGEnergyCorrector.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
class GBRWrapperRcd;

class EGEnergyCorrectorFactoryFromEventSetup {
public:
  EGEnergyCorrectorFactoryFromEventSetup(edm::ConsumesCollector, std::string const& regweigths);

  EGEnergyCorrectorFactoryFromEventSetup(const EGEnergyCorrectorFactoryFromEventSetup&) = delete;  // stop default
  const EGEnergyCorrectorFactoryFromEventSetup& operator=(const EGEnergyCorrectorFactoryFromEventSetup&) =
      delete;  // stop default

  // ---------- const member functions ---------------------
  EGEnergyCorrector::Initializer build(edm::EventSetup const&) const;

private:
  // ---------- member data --------------------------------
  edm::ESGetToken<GBRForest, GBRWrapperRcd> const ebCorrectionTag_;
  edm::ESGetToken<GBRForest, GBRWrapperRcd> const ebUncertaintyTag_;
  edm::ESGetToken<GBRForest, GBRWrapperRcd> const eeCorrectionTag_;
  edm::ESGetToken<GBRForest, GBRWrapperRcd> const eeUncertaintyTag_;
};

#endif
