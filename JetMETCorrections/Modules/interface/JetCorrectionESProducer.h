#ifndef JetCorrectionESProducer_h
#define JetCorrectionESProducer_h

//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006 (originally JetCorrectionService, renamed in 2011)
//

#include <string>
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

class JetCorrector;

#define DEFINE_JET_CORRECTION_ESPRODUCER(corrector_, name_ ) \
typedef JetCorrectionESProducer <corrector_>  name_; \
DEFINE_FWK_EVENTSETUP_MODULE(name_)

template <class Corrector>
class JetCorrectionESProducer : public edm::ESProducer
{
private:
  edm::ParameterSet mParameterSet;
  std::string mLevel;
  std::string mAlgo;

public:
  JetCorrectionESProducer(edm::ParameterSet const& fConfig) : mParameterSet(fConfig) 
  {
    std::string label = fConfig.getParameter<std::string>("@module_label"); 
    mLevel            = fConfig.getParameter<std::string>("level");
    mAlgo             = fConfig.getParameter<std::string>("algorithm");
        
    setWhatProduced(this, label);
  }

  ~JetCorrectionESProducer() {}

  boost::shared_ptr<JetCorrector> produce(JetCorrectionsRecord const& iRecord) 
  {
    edm::ESHandle<JetCorrectorParametersCollection> JetCorParColl;
    iRecord.get(mAlgo,JetCorParColl); 
    JetCorrectorParameters const& JetCorPar = (*JetCorParColl)[mLevel];
    boost::shared_ptr<JetCorrector> mCorrector(new Corrector(JetCorPar, mParameterSet));
    return mCorrector;
  }
};
#endif
