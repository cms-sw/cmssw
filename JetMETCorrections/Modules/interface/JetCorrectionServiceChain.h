//
// Original Author:  Fedor Ratnikov
//         Created:  Feb. 13, 2008
// $Id: JetCorrectionServiceChain.h,v 1.1 2009/09/24 13:18:55 bainbrid Exp $
//
//

// system include files
#include <string>
#include <vector>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

class JetCorrectionsRecord;

class JetCorrectionServiceChain : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
 public:
  JetCorrectionServiceChain (const edm::ParameterSet& fParameters);
  ~JetCorrectionServiceChain ();
  
  boost::shared_ptr<JetCorrector> produce (const JetCorrectionsRecord& );
  
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&, 
		      const edm::IOVSyncValue&, 
		      edm::ValidityInterval& fIOV);
  
 private:
  std::vector <std::string> mCorrectors;
  boost::shared_ptr<JetCorrector> mChainCorrector;
};
