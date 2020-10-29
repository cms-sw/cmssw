// -*- C++ -*-
//
// Package:    RecoEcal/EgammaCoreTools
// Class:      EcalMustacheSCParametersESProducer
//
/**\class EcalMustacheSCParametersESProducer

 Description: Produces the mustache superclusedr parameters

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Thomas Reis <Thomas.Reis@stfc.ac.uk>
//         Created:  Wed, 21 Oct 2020 15:05:26 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/DataRecord/interface/EcalMustacheSCParametersRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/MustacheSCParametersHelper.h"

//
// class declaration
//

class EcalMustacheSCParametersESProducer : public edm::ESProducer {
public:
  EcalMustacheSCParametersESProducer(const edm::ParameterSet&);
  ~EcalMustacheSCParametersESProducer() override;

  using ReturnType = std::unique_ptr<EcalMustacheSCParameters>;

  ReturnType produce(const EcalMustacheSCParametersRcd&);

private:
  EcalMustacheSCParameters params_;
};

//
// constructors and destructor
//
EcalMustacheSCParametersESProducer::EcalMustacheSCParametersESProducer(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);

  reco::MustacheSCParametersHelper mustacheSCParams(iConfig);
  params_ = static_cast<EcalMustacheSCParameters>(mustacheSCParams);
}

EcalMustacheSCParametersESProducer::~EcalMustacheSCParametersESProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
EcalMustacheSCParametersESProducer::ReturnType EcalMustacheSCParametersESProducer::produce(
    const EcalMustacheSCParametersRcd& iRecord) {
  auto product = std::make_unique<EcalMustacheSCParameters>(params_);
  return product;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalMustacheSCParametersESProducer);
