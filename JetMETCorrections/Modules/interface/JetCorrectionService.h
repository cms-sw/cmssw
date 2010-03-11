#ifndef JetCorrectionService_h
#define JetCorrectionService_h

//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006
// $Id: JetCorrectionService.h,v 1.3 2010/02/25 23:09:09 wmtan Exp $
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

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
  std::string level;
  std::string label;
  std::string algorithm;
  std::string name;
  std::string section;
  std::string payload_name;
  // construction / destruction
public:
  JetCorrectionService(const edm::ParameterSet& fParameters) : 
    level(fParameters.getParameter<std::string>("level")),
    label(fParameters.getParameter<std::string>("label")),
    algorithm(fParameters.getParameter<std::string>("algorithm")),
    name(fParameters.getParameter<std::string>("@module_label")),
    section(fParameters.getParameter<std::string>("section"))
  {
    setWhatProduced(this, name);
    findingRecord <JetCorrectionsRecord> ();
    // derive the overall string to look for
    payload_name = level;
    if (!algorithm.empty()) payload_name += "_" + algorithm + "_" + section;
  }
  
  ~JetCorrectionService () {}
  
  // member functions
  boost::shared_ptr<JetCorrector> produce(const JetCorrectionsRecord& iRecord) {
    edm::ESHandle<JetCorrectorParameters> params;
    iRecord.get(payload_name,params);    
    boost::shared_ptr<JetCorrector> theCorrector(new Corrector(*params, level)); 
    return theCorrector;
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
