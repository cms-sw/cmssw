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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CondFormats/DataRecord/interface/EcalMustacheSCParametersRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/MustacheSCParametersHelper.h"

//
// class declaration
//

class EcalMustacheSCParametersESProducer : public edm::ESProducer {
public:
  EcalMustacheSCParametersESProducer(const edm::ParameterSet&);
  ~EcalMustacheSCParametersESProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

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

  reco::MustacheSCParametersHelper mustacheSCParams(params_, iConfig);
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

void EcalMustacheSCParametersESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("sqrtLogClustETuning", 1.1);
  {
    edm::ParameterSetDescription vpsd1;
    vpsd1.add<double>("log10EMin");
    vpsd1.add<double>("etaMin");
    vpsd1.add<std::vector<double>>("pUp");
    vpsd1.add<std::vector<double>>("pLow");
    vpsd1.add<std::vector<double>>("w0Up");
    vpsd1.add<std::vector<double>>("w1Up");
    vpsd1.add<std::vector<double>>("w0Low");
    vpsd1.add<std::vector<double>>("w1Low");
    std::vector<edm::ParameterSet> temp1;
    temp1.reserve(1);
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("log10EMin", -3.);
      temp2.addParameter<double>("etaMin", 0.);
      temp2.addParameter<std::vector<double>>("pUp", {-0.107537, 0.590969, -0.076494});
      temp2.addParameter<std::vector<double>>("pLow", {-0.0268843, 0.147742, -0.0191235});
      temp2.addParameter<std::vector<double>>("w0Up", {-0.00681785, -0.00239516});
      temp2.addParameter<std::vector<double>>("w1Up", {0.000699995, -0.00554331});
      temp2.addParameter<std::vector<double>>("w0Low", {-0.00681785, -0.00239516});
      temp2.addParameter<std::vector<double>>("w1Low", {0.000699995, -0.00554331});
      temp1.push_back(temp2);
    }
    desc.addVPSet("parabolaParameterSets", vpsd1, temp1);
  }
  descriptions.add("ecalMustacheSCParametersESProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalMustacheSCParametersESProducer);
