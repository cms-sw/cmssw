#ifndef JetCorrectionService_h
#define JetCorrectionService_h

//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006
// $Id: JetCorrectionService.icc,v 1.1 2007/01/18 01:35:12 fedor Exp $
//
//

// system include files
#include <memory>
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

// macro to define instance of JetCorrectionService
#define DEFINE_JET_CORRECTION_SERVICE(corrector_, name_ ) \
typedef JetCorrectionService <corrector_>  name_; \
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(name_)

// Correction Service itself
template <class Corrector>
class JetCorrectionService : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
private:
  boost::shared_ptr<JetCorrector> mCorrector;
public:
  JetCorrectionService (const edm::ParameterSet& fParameters) 
    : mCorrector (new Corrector (fParameters))
  {
    std::string label = fParameters.template getParameter <std::string> ("label");
    setWhatProduced(this, label);
    findingRecord <JetCorrectionsRecord> ();
  }
  
  ~JetCorrectionService () {}

  boost::shared_ptr<JetCorrector> produce( const JetCorrectionsRecord& ) {
    return mCorrector;
  }
  
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, 
		      const edm::IOVSyncValue&, 
		      edm::ValidityInterval& fIOV) {
    fIOV = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime()); // anytime
  }
};

#endif
