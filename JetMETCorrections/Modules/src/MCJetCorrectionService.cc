//
// Original Author:  Fedor Ratnikov
//         Created:  Dec. 28, 2006
// $Id: MCJetCorrectionService.cc,v 1.1 2006/12/29 00:48:40 fedor Exp $
//
//

#include "MCJetCorrectionService.h"

#include "JetMETCorrections/MCJet/interface/MCJetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"


MCJetCorrectionService::MCJetCorrectionService (const edm::ParameterSet& fParameters) 
  : mCorrector (new MCJetCorrector (fParameters))
{
  std::string label = fParameters.getParameter <std::string> ("label");
  setWhatProduced(this, label);
  findingRecord <JetCorrectionsRecord> ();
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

