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
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

#include "CondFormats/DataRecord/interface/EcalSCDynamicDPhiParametersRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/SCDynamicDPhiParametersHelper.h"

//
// class declaration
//

class EcalSCDynamicDPhiParametersESProducer : public edm::ESProducer {
public:
  EcalSCDynamicDPhiParametersESProducer(const edm::ParameterSet&);
  ~EcalSCDynamicDPhiParametersESProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

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

  reco::SCDynamicDPhiParametersHelper scDynamicDPhiParams(params_, iConfig);
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

void EcalSCDynamicDPhiParametersESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription vpsd1;
    vpsd1.add<double>("eMin");
    vpsd1.add<double>("etaMin");
    vpsd1.add<double>("yoffset");
    vpsd1.add<double>("scale");
    vpsd1.add<double>("xoffset");
    vpsd1.add<double>("width");
    vpsd1.add<double>("saturation");
    vpsd1.add<double>("cutoff");
    std::vector<edm::ParameterSet> temp1;
    temp1.reserve(1);
    {
      edm::ParameterSet temp2;
      temp2.addParameter<double>("eMin", 0.);
      temp2.addParameter<double>("etaMin", 0.);
      temp2.addParameter<double>("yoffset", 0.0280506);
      temp2.addParameter<double>("scale", 0.946048);
      temp2.addParameter<double>("xoffset", -0.101172);
      temp2.addParameter<double>("width", 0.432767);
      temp2.addParameter<double>("saturation", 0.14);
      temp2.addParameter<double>("cutoff", 0.6);
      temp1.push_back(temp2);
    }
    desc.addVPSet("dynamicDPhiParameterSets", vpsd1, temp1);
  }
  descriptions.add("ecalSCDynamicDPhiParametersESProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalSCDynamicDPhiParametersESProducer);
