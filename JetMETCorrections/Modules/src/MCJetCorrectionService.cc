//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006
// $Id: HcalDbProducer.h,v 1.9 2006/10/18 23:37:50 fedor Exp $
//
//

#include "MCJetCorrectionService.h"

#include "JetMETCorrections/MCJet/interface/MCJetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"


MCJetCorrectionService::MCJetCorrectionService (const edm::ParameterSet& fParameters) 
  : mCorrector (new MCJetCorrector (fParameters))
{
  setWhatProduced(this);
}

MCJetCorrectionService::~MCJetCorrectionService () {}

boost::shared_ptr<JetCorrector> MCJetCorrectionService::produce( const JetCorrectionsRecord& ) {
  return mCorrector;
}

void MCJetCorrectionService::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, 
					  const edm::IOVSyncValue&, 
					  edm::ValidityInterval& fIOV) {
  fIOV = edm::ValidityInterval(edm::IOVSyncValue::beginOfTime(),edm::IOVSyncValue::endOfTime()); // anytime
}

