#ifndef JetCorrectionESChain_h
#define JetCorrectionESChain_h

//
// Original Author:  Fedor Ratnikov
//         Created:  Feb. 13, 2008
//         (originally named JetCorrectionServiceChain, renamed in 2011)
// $Id: JetCorrectionESChain.h,v 1.1.2.1 2011/10/17 20:54:17 wdd Exp $
//
//

#include <string>
#include <vector>
#include "boost/shared_ptr.hpp"

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

  boost::shared_ptr<JetCorrector> produce(JetCorrectionsRecord const& );

private:
  std::vector <std::string> mCorrectors;
  boost::shared_ptr<JetCorrector> mChainCorrector;
};
#endif
