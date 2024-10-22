//
// Toyoko Orimoto (Caltech), 10 July 2007
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbRecord.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

class EcalLaserDbAnalyzer : public edm::global::EDAnalyzer<> {
public:
  explicit EcalLaserDbAnalyzer(const edm::ParameterSet&);
  ~EcalLaserDbAnalyzer() override = default;

  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

private:
  // ----------member data ---------------------------
  const edm::ESGetToken<EcalLaserDbService, EcalLaserDbRecord> laserDbToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalLaserDbAnalyzer::EcalLaserDbAnalyzer(const edm::ParameterSet& iConfig) : laserDbToken_(esConsumes()) {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void EcalLaserDbAnalyzer::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const {
  // get record from offline DB
  const auto& setup = iSetup.getData(laserDbToken_);
  edm::LogInfo("EcalLaserDbService") << "EcalLaserDbAnalyzer::analyze-> got EcalLaserDbRecord:";

  EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdpnpair;
  const EcalLaserAPDPNRatios* myapdpn = setup.getAPDPNRatios();
  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = myapdpn->getLaserMap();

  //   EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp;
  //   const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap =  myapdpn->getTimeMap();

  EcalLaserAPDPNref apdpnref;
  const EcalLaserAPDPNRatiosRef* myapdpnref = setup.getAPDPNRatiosRef();
  const EcalLaserAPDPNRatiosRefMap& laserRefMap = myapdpnref->getMap();

  EcalLaserAlpha alpha;
  const EcalLaserAlphas* myalpha = setup.getAlphas();
  const EcalLaserAlphaMap& laserAlphaMap = myalpha->getMap();

  EcalLinearCorrections::Values linValues;
  const EcalLinearCorrections* mylinear = setup.getLinearCorrections();
  const EcalLinearCorrections::EcalValueMap& linearValueMap = mylinear->getValueMap();

  for (int ieta = -EBDetId::MAX_IETA; ieta <= EBDetId::MAX_IETA; ++ieta) {
    if (ieta == 0)
      continue;
    for (int iphi = EBDetId::MIN_IPHI; iphi <= EBDetId::MAX_IPHI; ++iphi) {
      EBDetId ebdetid(ieta, iphi);

      edm::LogVerbatim("EcalLaserDbService") << ebdetid << " " << ebdetid.ietaSM() << " " << ebdetid.iphiSM() << "\n";

      EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap::const_iterator itratio = laserRatiosMap.find(ebdetid);
      if (itratio != laserRatiosMap.end()) {
        apdpnpair = (*itratio);
        edm::LogVerbatim("EcalLaserDbService")
            << " APDPN pair = " << apdpnpair.p1 << " , " << apdpnpair.p2 << " , " << apdpnpair.p3 << "\n";
      } else {
        edm::LogError("EcalLaserDbService") << "error with laserRatiosMap!";
      }

      EcalLinearCorrections::EcalValueMap::const_iterator itlin = linearValueMap.find(ebdetid);
      if (itlin != linearValueMap.end()) {
        linValues = (*itlin);
        edm::LogVerbatim("EcalLaserDbService")
            << " APDPN pair = " << linValues.p1 << " , " << linValues.p2 << " , " << linValues.p3 << "\n";
      } else {
        edm::LogError("EcalLaserDbService") << "error with linearValuesMap!";
      }

      EcalLaserAPDPNRatiosRefMap::const_iterator itref = laserRefMap.find(ebdetid);
      if (itref != laserRefMap.end()) {
        apdpnref = (*itref);
        edm::LogVerbatim("EcalLaserDbService") << " APDPN ref = " << apdpnref << "\n";
      } else {
        edm::LogError("EcalLaserDbService") << "error with laserRefMap!";
      }

      EcalLaserAlphaMap::const_iterator italpha = laserAlphaMap.find(ebdetid);
      if (italpha != laserAlphaMap.end()) {
        alpha = (*italpha);
        edm::LogVerbatim("EcalLaserDbService") << " ALPHA = " << alpha << "\n";
      } else {
        edm::LogError("EcalLaserDbService") << "error with laserAlphaMap!";
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(EcalLaserDbAnalyzer);
