#ifndef JetCorrectionESProducer_h
#define JetCorrectionESProducer_h

//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006 (originally JetCorrectionService, renamed in 2011)
//

#include <string>
#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

class JetCorrector;

#define DEFINE_JET_CORRECTION_ESPRODUCER(corrector_, name_) \
  typedef JetCorrectionESProducer<corrector_> name_;        \
  DEFINE_FWK_EVENTSETUP_MODULE(name_)

template <class Corrector>
class JetCorrectionESProducer : public edm::ESProducer {
private:
  const edm::ParameterSet mParameterSet;
  const std::string mLevel;
  const edm::ESGetToken<JetCorrectorParametersCollection, JetCorrectionsRecord> mToken;

public:
  JetCorrectionESProducer(edm::ParameterSet const& fConfig)
      : mParameterSet(fConfig),
        mLevel(fConfig.getParameter<std::string>("level")),
        mToken(setWhatProduced(this, fConfig.getParameter<std::string>("@module_label"))
                   .consumes(edm::ESInputTag{"", fConfig.getParameter<std::string>("algorithm")})) {}

  ~JetCorrectionESProducer() override {}

  std::unique_ptr<JetCorrector> produce(JetCorrectionsRecord const& iRecord) {
    const auto& jetCorParColl = iRecord.get(mToken);
    return std::make_unique<Corrector>(jetCorParColl[mLevel], mParameterSet);
  }
};
#endif
