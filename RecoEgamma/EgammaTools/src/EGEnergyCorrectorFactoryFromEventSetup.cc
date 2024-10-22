// -*- C++ -*-
//
// Package:     RecoEgamma/EgammaTools
// Class  :     EGEnergyCorrectorFactoryFromEventSetup
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 03 Sep 2021 18:55:05 GMT
//

// system include files

// user include files
#include "RecoEgamma/EgammaTools/interface/EGEnergyCorrectorFactoryFromEventSetup.h"

#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EGEnergyCorrectorFactoryFromEventSetup::EGEnergyCorrectorFactoryFromEventSetup(edm::ConsumesCollector cc,
                                                                               std::string const& regweigths)
    : ebCorrectionTag_(cc.esConsumes(edm::ESInputTag("", regweigths + "_EBCorrection"))),
      ebUncertaintyTag_(cc.esConsumes(edm::ESInputTag("", regweigths + "_EBUncertainty"))),
      eeCorrectionTag_(cc.esConsumes(edm::ESInputTag("", regweigths + "_EECorrection"))),
      eeUncertaintyTag_(cc.esConsumes(edm::ESInputTag("", regweigths + "_EEUncertainty"))) {}

//
// const member functions
//
EGEnergyCorrector::Initializer EGEnergyCorrectorFactoryFromEventSetup::build(edm::EventSetup const& iSetup) const {
  EGEnergyCorrector::Initializer ret;
  auto no_del = [](void const*) {};

  ret.readereb_ = std::shared_ptr<GBRForest const>(&iSetup.getData(ebCorrectionTag_), no_del);
  ret.readerebvariance_ = std::shared_ptr<GBRForest const>(&iSetup.getData(ebUncertaintyTag_), no_del);
  ret.readeree_ = std::shared_ptr<GBRForest const>(&iSetup.getData(eeCorrectionTag_), no_del);
  ret.readereevariance_ = std::shared_ptr<GBRForest const>(&iSetup.getData(eeUncertaintyTag_), no_del);

  return ret;
}

//
// static member functions
//
