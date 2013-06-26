//
// Toyoko Orimoto (Caltech), 10 July 2007
//

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

class EcalLaserDbService;
class EcalLaserDbRecord;

#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"


class EcalLaserCorrectionService : public edm::ESProducer {
public:
  EcalLaserCorrectionService( const edm::ParameterSet& );
  ~EcalLaserCorrectionService();
  
  boost::shared_ptr<EcalLaserDbService> produce( const EcalLaserDbRecord& );
  
  // callbacks
  void alphaCallback (const EcalLaserAlphasRcd& fRecord);
  void apdpnRefCallback (const EcalLaserAPDPNRatiosRefRcd& fRecord);
  void apdpnCallback (const EcalLaserAPDPNRatiosRcd& fRecord);
  void linearCallback (const EcalLinearCorrectionsRcd& fRecord);
  
private:
  // ----------member data ---------------------------
  boost::shared_ptr<EcalLaserDbService> mService_;
  //  std::vector<std::string> mDumpRequest;
  //  std::ostream* mDumpStream;
};
