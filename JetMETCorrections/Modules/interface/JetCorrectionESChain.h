#ifndef JetCorrectionESChain_h
#define JetCorrectionESChain_h

//
// Original Author:  Fedor Ratnikov
//         Created:  Feb. 13, 2008
//         (originally named JetCorrectionServiceChain, renamed in 2011)
//
//

#include <string>
#include <memory>
#include <vector>

#include "FWCore/Framework/interface/ESProducer.h"

class JetCorrectionsRecord;
class JetCorrector;
namespace edm {
  class ParameterSet;
}

class JetCorrectionESChain : public edm::ESProducer {
public:
  JetCorrectionESChain(edm::ParameterSet const& fParameters);
  ~JetCorrectionESChain();

  std::shared_ptr<JetCorrector> produce(JetCorrectionsRecord const& );

private:
  std::vector <std::string> mCorrectors;
  std::shared_ptr<JetCorrector> mChainCorrector;
};
#endif
