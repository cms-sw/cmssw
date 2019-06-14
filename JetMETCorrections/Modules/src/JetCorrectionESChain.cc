//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006
//
//

#include "JetMETCorrections/Modules/interface/JetCorrectionESChain.h"
#include "JetMETCorrections/Objects/interface/ChainedJetCorrector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include <algorithm>

JetCorrectionESChain::JetCorrectionESChain(edm::ParameterSet const& fParameters)
    : mCorrectors(fParameters.getParameter<std::vector<std::string> >("correctors")) {
  std::string label(fParameters.getParameter<std::string>("@module_label"));
  if (std::find(mCorrectors.begin(), mCorrectors.end(), label) != mCorrectors.end()) {
    throw cms::Exception("Recursion is not allowed")
        << "JetCorrectionESChain: corrector " << label << " is chained to itself";
  }
  setWhatProduced(this, label);
}

JetCorrectionESChain::~JetCorrectionESChain() {}

std::unique_ptr<JetCorrector> JetCorrectionESChain::produce(JetCorrectionsRecord const& fRecord) {
  std::unique_ptr<ChainedJetCorrector> corrector{new ChainedJetCorrector};
  corrector->clear();
  for (size_t i = 0; i < mCorrectors.size(); ++i) {
    edm::ESHandle<JetCorrector> handle;
    fRecord.get(mCorrectors[i], handle);
    corrector->push_back(&*handle);
  }
  return corrector;
}
