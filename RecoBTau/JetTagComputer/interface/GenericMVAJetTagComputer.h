#ifndef RecoBTau_JetTagComputer_GenericMVAJetTagComputer_h
#define RecoBTau_JetTagComputer_GenericMVAJetTagComputer_h

#include <memory>

#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/GenericMVAComputerCache.h"
#include "RecoBTau/JetTagComputer/interface/TagInfoMVACategorySelector.h"

class JetTagComputerRecord;

class GenericMVAJetTagComputer : public JetTagComputer {
public:
  struct Tokens {
    Tokens(const edm::ParameterSet &parameters, edm::ESConsumesCollector &&cc);
    edm::ESGetToken<PhysicsTools::Calibration::MVAComputerContainer, BTauGenericMVAJetTagComputerRcd> calib_;
  };

  GenericMVAJetTagComputer(const edm::ParameterSet &parameters, Tokens tokens);
  ~GenericMVAJetTagComputer() override;

  void initialize(const JetTagComputerRecord &) override;

  float discriminator(const TagInfoHelper &info) const override;

  virtual reco::TaggingVariableList taggingVariables(const reco::BaseTagInfo &tagInfo) const;
  virtual reco::TaggingVariableList taggingVariables(const TagInfoHelper &info) const;

private:
  std::unique_ptr<TagInfoMVACategorySelector> categorySelector_;
  GenericMVAComputerCache computerCache_;
  Tokens tokens_;
};

#endif  // RecoBTau_JetTagComputer_GenericMVAJetTagComputer_h
