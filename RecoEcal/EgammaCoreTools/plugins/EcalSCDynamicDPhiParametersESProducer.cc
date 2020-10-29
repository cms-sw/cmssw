// -*- C++ -*-
//
// Package:    RecoEcal/EgammaCoreTools
// Class:      EcalSCDynamicDPhiParametersESProducer
//
/**\class EcalSCDynamicDPhiParametersESProducer

 Description: Produces the supercluster dynamic dPhi parameters

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Reis <Thomas.Reis@stfc.ac.uk>
//         Created:  Wed, 28 Oct 2020 16:17:26 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/DataRecord/interface/EcalSCDynamicDPhiParametersRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/SCDynamicDPhiParametersHelper.h"

//
// class declaration
//

class EcalSCDynamicDPhiParametersESProducer : public edm::ESProducer {
public:
  EcalSCDynamicDPhiParametersESProducer(const edm::ParameterSet&);
  ~EcalSCDynamicDPhiParametersESProducer() override;

  using ReturnType = std::unique_ptr<EcalSCDynamicDPhiParameters>;

  ReturnType produce(const EcalSCDynamicDPhiParametersRcd&);

private:
  EcalSCDynamicDPhiParameters params_;
};

//
// constructors and destructor
//
EcalSCDynamicDPhiParametersESProducer::EcalSCDynamicDPhiParametersESProducer(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);

  reco::SCDynamicDPhiParametersHelper scDynamicDPhiParams(iConfig);
  params_ = static_cast<EcalSCDynamicDPhiParameters>(scDynamicDPhiParams);
}

EcalSCDynamicDPhiParametersESProducer::~EcalSCDynamicDPhiParametersESProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
EcalSCDynamicDPhiParametersESProducer::ReturnType EcalSCDynamicDPhiParametersESProducer::produce(
    const EcalSCDynamicDPhiParametersRcd& iRecord) {
  auto product = std::make_unique<EcalSCDynamicDPhiParameters>(params_);
  return product;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalSCDynamicDPhiParametersESProducer);
