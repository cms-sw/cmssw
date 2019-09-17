//
// Toyoko Orimoto (Caltech), 10 July 2007
// Andrea Massironi, 3 Aug 2019
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"

class EcalLaserDbServiceMC;
class EcalLaserDbRecordMC;

#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosMCRcd.h"
#include "CondFormats/DataRecord/interface/EcalLinearCorrectionsRcd.h"

class EcalLaserCorrectionServiceMC : public edm::ESProducer {
public:
  EcalLaserCorrectionServiceMC(const edm::ParameterSet&);
  ~EcalLaserCorrectionServiceMC() override;

  std::shared_ptr<EcalLaserDbServiceMC> produce(const EcalLaserDbRecordMC&);

private:
  using HostType = edm::ESProductHost<EcalLaserDbServiceMC,
                                      EcalLaserAlphasRcd,
                                      EcalLaserAPDPNRatiosRefRcd,
                                      EcalLaserAPDPNRatiosMCRcd,
                                      EcalLinearCorrectionsRcd>;

  void setupAlpha(const EcalLaserAlphasRcd& fRecord, EcalLaserDbServiceMC& service);
  void setupApdpnRef(const EcalLaserAPDPNRatiosRefRcd& fRecord, EcalLaserDbServiceMC& service);
  void setupApdpn(const EcalLaserAPDPNRatiosMCRcd& fRecord, EcalLaserDbServiceMC& service);
  void setupLinear(const EcalLinearCorrectionsRcd& fRecord, EcalLaserDbServiceMC& service);

  // ----------member data ---------------------------
  edm::ReusableObjectHolder<HostType> holder_;

  //  std::vector<std::string> mDumpRequest;
  //  std::ostream* mDumpStream;
};
