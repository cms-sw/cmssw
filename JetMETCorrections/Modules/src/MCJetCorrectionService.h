//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 27, 2006
// $Id: HcalDbProducer.h,v 1.9 2006/10/18 23:37:50 fedor Exp $
//
//

#ifndef MCJetCorrectionService_h
#define MCJetCorrectionService_h

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

class JetCorrector;
class JetCorrectionsRecord;

class MCJetCorrectionService : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
 public:
  MCJetCorrectionService (const edm::ParameterSet& fParameters);
  virtual ~MCJetCorrectionService ();
  boost::shared_ptr<JetCorrector> produce( const JetCorrectionsRecord& );
 private:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval& );
  boost::shared_ptr<JetCorrector> mCorrector;
};

#endif
