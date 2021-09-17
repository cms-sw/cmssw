#include "RecoTauTag/RecoTau/interface/TauDiscriminationProducerBase.h"
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

/* 
 * class PFRecoTauDiscriminatioLogicalAndProducer
 *
 * Applies a boolean operator (AND or OR) to a set 
 * of PFTauDiscriminators.  Note that the set of PFTauDiscriminators
 * is accessed/combined in the base class (the Prediscriminants).
 *
 * This class merely exposes this behavior directly by 
 * returning true for all taus that pass the prediscriminants
 *
 * revised : Mon Aug 31 12:59:50 PDT 2009
 * Authors : Michele Pioppi, Evan Friis (UC Davis)
 */

using namespace reco;

class PFTauDiscriminatorLogicalAndProducer : public PFTauDiscriminationProducerBase {
public:
  explicit PFTauDiscriminatorLogicalAndProducer(const edm::ParameterSet&);
  ~PFTauDiscriminatorLogicalAndProducer() override{};
  double discriminate(const PFTauRef& pfTau) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  double passResult_;
};

PFTauDiscriminatorLogicalAndProducer::PFTauDiscriminatorLogicalAndProducer(const edm::ParameterSet& iConfig)
    : PFTauDiscriminationProducerBase(iConfig) {
  passResult_ = iConfig.getParameter<double>("PassValue");
  prediscriminantFailValue_ = iConfig.getParameter<double>("FailValue");  //defined in base class
}

double PFTauDiscriminatorLogicalAndProducer::discriminate(const PFTauRef& pfTau) const {
  // if this function is called on a tau, it is has passed (in the base class)
  // the set of prediscriminants, using the prescribed boolean operation.  thus
  // we only need to return TRUE
  return passResult_;
}

void PFTauDiscriminatorLogicalAndProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // PFTauDiscriminatorLogicalAndProducer
  edm::ParameterSetDescription desc;

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("BooleanOperator", "and");
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut", 0.5);
      psd1.add<edm::InputTag>("Producer", edm::InputTag("pfRecoTauDiscriminationAgainstElectron"));
      psd0.add<edm::ParameterSetDescription>("discr2", psd1);
    }
    {
      edm::ParameterSetDescription psd1;
      psd1.add<double>("cut", 0.5);
      psd1.add<edm::InputTag>("Producer", edm::InputTag("pfRecoTauDiscriminationByIsolation"));
      psd0.add<edm::ParameterSetDescription>("discr1", psd1);
    }
    desc.add<edm::ParameterSetDescription>("Prediscriminants", psd0);
  }

  desc.add<double>("PassValue", 1.0);
  desc.add<double>("FailValue", 0.0);
  desc.add<edm::InputTag>("PFTauProducer", edm::InputTag("pfRecoTauProducer"));
  descriptions.add("PFTauDiscriminatorLogicalAndProducer", desc);
}

DEFINE_FWK_MODULE(PFTauDiscriminatorLogicalAndProducer);
