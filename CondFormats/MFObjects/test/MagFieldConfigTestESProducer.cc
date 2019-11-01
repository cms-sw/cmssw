// -*- C++ -*-
//
// Package:    CondTools/RunInfo
// Class:      MagFieldConfigTestESProducer
//
/**\class MagFieldConfigTestESProducer

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 02 Oct 2019 17:34:35 GMT
//
//

// system include files
#include <memory>
#include <unordered_map>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CondFormats/MFObjects/interface/MagFieldConfig.h"
#include "CondFormats/DataRecord/interface/MagFieldConfigRcd.h"

//
// class declaration
//

class MagFieldConfigTestESProducer : public edm::ESProducer {
public:
  MagFieldConfigTestESProducer(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<MagFieldConfig>;

  ReturnType produce(const MagFieldConfigRcd&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::pair<unsigned int, MagFieldConfig> makeMagFieldConfig(edm::ParameterSet const& pset) const;
  // ----------member data ---------------------------
  std::unordered_map<unsigned int, MagFieldConfig> configs_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MagFieldConfigTestESProducer::MagFieldConfigTestESProducer(const edm::ParameterSet& iConfig) {
  std::vector<edm::ParameterSet> const& configs = iConfig.getParameter<std::vector<edm::ParameterSet>>("configs");
  configs_.reserve(configs.size());
  for (auto const& pset : configs) {
    configs_.insert(makeMagFieldConfig(pset));
  }

  setWhatProduced(this);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
MagFieldConfigTestESProducer::ReturnType MagFieldConfigTestESProducer::produce(const MagFieldConfigRcd& iRecord) {
  const unsigned int run = iRecord.validityInterval().first().eventID().run();

  auto itFound = configs_.find(run);
  if (itFound == configs_.end()) {
    return nullptr;
  }
  return std::make_unique<MagFieldConfig>(itFound->second);
}

std::pair<unsigned int, MagFieldConfig> MagFieldConfigTestESProducer::makeMagFieldConfig(
    edm::ParameterSet const& pset) const {
  return std::pair<unsigned int, MagFieldConfig>(pset.getParameter<unsigned int>("run"),
                                                 pset.getParameter<edm::ParameterSet>("config"));
}

void MagFieldConfigTestESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription entryDesc;
    entryDesc.add<unsigned int>("run");
    {
      //Do not enforce what the MagFieldConfig wants
      edm::ParameterSetDescription magConfig;
      magConfig.setAllowAnything();
      entryDesc.add<edm::ParameterSetDescription>("config", magConfig);
    }

    desc.addVPSet("configs", entryDesc, {});
  }
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(MagFieldConfigTestESProducer);
