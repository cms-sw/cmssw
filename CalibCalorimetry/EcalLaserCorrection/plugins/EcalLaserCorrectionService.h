//
// Toyoko Orimoto (Caltech), 10 July 2007
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

class EcalLaserDbService;
class EcalLaserDbRecord;

#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"


class EcalLaserCorrectionService : public edm::ESProducer {
public:
  EcalLaserCorrectionService( const edm::ParameterSet& );
  ~EcalLaserCorrectionService() override;
  
  std::shared_ptr<EcalLaserDbService> produce( const EcalLaserDbRecord& );

private:

  using HostType = edm::ESProductHost<EcalLaserDbService,
                                      EcalLaserAlphasRcd,
                                      EcalLaserAPDPNRatiosRefRcd,
                                      EcalLaserAPDPNRatiosRcd,
                                      EcalLinearCorrectionsRcd>;

  void setupAlpha(const EcalLaserAlphasRcd& fRecord,
                  EcalLaserDbService& service);
  void setupApdpnRef(const EcalLaserAPDPNRatiosRefRcd& fRecord,
                     EcalLaserDbService& service);
  void setupApdpn(const EcalLaserAPDPNRatiosRcd& fRecord,
                  EcalLaserDbService& service);
  void setupLinear(const EcalLinearCorrectionsRcd& fRecord,
                   EcalLaserDbService& service);

  // ----------member data ---------------------------
  edm::ReusableObjectHolder<HostType> holder_;

  //  std::vector<std::string> mDumpRequest;
  //  std::ostream* mDumpStream;
};
