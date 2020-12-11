//
// Original Author:  Fedor Ratnikov
//         Created:  Feb. 13, 2008
//         (originally named JetCorrectionServiceChain, renamed in 2011)
//
//

#include <string>
#include <memory>
#include <vector>
#include <algorithm>

#include "JetMETCorrections/Objects/interface/ChainedJetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

class JetCorrectionESChain : public edm::ESProducer {
public:
  JetCorrectionESChain(edm::ParameterSet const& fParameters);
  ~JetCorrectionESChain() override;

  std::unique_ptr<JetCorrector> produce(JetCorrectionsRecord const&);

private:
  std::vector<edm::ESGetToken<JetCorrector, JetCorrectionsRecord>> tokens_;
};

JetCorrectionESChain::JetCorrectionESChain(edm::ParameterSet const& fParameters) {
  std::string label(fParameters.getParameter<std::string>("@module_label"));
  auto correctors = fParameters.getParameter<std::vector<std::string>>("correctors");
  if (std::find(correctors.begin(), correctors.end(), label) != correctors.end()) {
    throw cms::Exception("Recursion is not allowed")
        << "JetCorrectionESChain: corrector " << label << " is chained to itself";
  }
  auto cc = setWhatProduced(this, label);
  tokens_.resize(correctors.size());
  for (size_t i = 0; i < correctors.size(); ++i) {
    tokens_[i] = cc.consumes(edm::ESInputTag{"", correctors[i]});
  }
}

JetCorrectionESChain::~JetCorrectionESChain() {}

std::unique_ptr<JetCorrector> JetCorrectionESChain::produce(JetCorrectionsRecord const& fRecord) {
  auto corrector = std::make_unique<ChainedJetCorrector>();
  for (const auto& token : tokens_) {
    corrector->push_back(&fRecord.get(token));
  }
  return corrector;
}

DEFINE_FWK_EVENTSETUP_MODULE(JetCorrectionESChain);
