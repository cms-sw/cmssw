#ifndef JetCorrectionService_h
#define JetCorrectionService_h

//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006
// $Id: JetCorrectionService.h,v 1.2 2009/11/12 18:08:28 schiefer Exp $
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
DEFINE_FWK_EVENTSETUP_SOURCE(name_)


// Correction Service itself
template <class Corrector>
class JetCorrectionService : public edm::ESProducer,
			     public edm::EventSetupRecordIntervalFinder
{
  // member data
private:
  boost::shared_ptr<JetCorrector> mCorrector;

  // construction / destruction
public:
  JetCorrectionService(const edm::ParameterSet& fParameters) 
    : mCorrector(new Corrector(fParameters))
  {
    std::string label(fParameters.getParameter<std::string>("@module_label"));
    
    setWhatProduced(this, label);
    findingRecord <JetCorrectionsRecord> ();
  }
  
  ~JetCorrectionService () {}
  
  // member functions
  boost::shared_ptr<JetCorrector> produce(const JetCorrectionsRecord&) {
    return mCorrector;
  }
  
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, 
		      const edm::IOVSyncValue&, 
		      edm::ValidityInterval& fIOV)
  {
    fIOV = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),
				 edm::IOVSyncValue::endOfTime()); // anytime
  }
};

#endif
