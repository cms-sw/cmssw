
/*----------------------------------------------------------------------

Sh. Rahatlou, University of Rome & INFN
simple analyzer to dump information about ECAL cond objects

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibErrorsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibErrorsRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/DataRecord/interface/EcalLaserAlphasRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/DataRecord/interface/EcalLaserAPDPNRatiosRefRcd.h"

#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"
#include "CondFormats/DataRecord/interface/EcalMappingElectronicsRcd.h"

using namespace std;

class EcalObjectAnalyzer : public edm::EDAnalyzer {
public:
  explicit EcalObjectAnalyzer(edm::ParameterSet const& p) {}
  explicit EcalObjectAnalyzer(int i) {}
  ~EcalObjectAnalyzer() override {}
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
};

void EcalObjectAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context) {
  using namespace edm::eventsetup;
  // Context is not used.
  std::cout << ">>> EcalObjectAnalyzer: processing run " << e.id().run() << " event: " << e.id().event() << std::endl;

  edm::ESHandle<EcalPedestals> pPeds;
  context.get<EcalPedestalsRcd>().get(pPeds);

  // ADC -> GeV Scale
  edm::ESHandle<EcalADCToGeVConstant> pAgc;
  context.get<EcalADCToGeVConstantRcd>().get(pAgc);
  const EcalADCToGeVConstant* agc = pAgc.product();
  std::cout << "Global ADC->GeV scale: EB " << agc->getEBValue() << " GeV/ADC count"
            << " EE " << agc->getEEValue() << " GeV/ADC count" << std::endl;

  const EcalPedestals* myped = pPeds.product();
  // Barrel loop
  int cnt = 0;
  for (EcalPedestals::const_iterator it = myped->barrelItems().begin(); it != myped->barrelItems().end(); ++it) {
    std::cout << "EcalPedestal: "
              << " BARREL " << cnt << " "
              << "  mean_x1:  " << (*it).mean_x1 << " rms_x1: " << (*it).rms_x1 << "  mean_x6:  " << (*it).mean_x6
              << " rms_x6: " << (*it).rms_x6 << "  mean_x12: " << (*it).mean_x12 << " rms_x12: " << (*it).rms_x12
              << std::endl;
    ++cnt;
  }
  // Endcap loop
  for (EcalPedestals::const_iterator it = myped->endcapItems().begin(); it != myped->endcapItems().end(); ++it) {
    std::cout << "EcalPedestal: "
              << " ENDCAP "
              << "  mean_x1:  " << (*it).mean_x1 << " rms_x1: " << (*it).rms_x1 << "  mean_x6:  " << (*it).mean_x6
              << " rms_x6: " << (*it).rms_x6 << "  mean_x12: " << (*it).mean_x12 << " rms_x12: " << (*it).rms_x12
              << std::endl;
  }

  // fetch map of groups of xtals
  edm::ESHandle<EcalWeightXtalGroups> pGrp;
  context.get<EcalWeightXtalGroupsRcd>().get(pGrp);
  const EcalWeightXtalGroups* grp = pGrp.product();
  EcalXtalGroupsMap::const_iterator git;
  // Barrel loop
  for (git = grp->barrelItems().begin(); git != grp->barrelItems().end(); ++git) {
    std::cout << "XtalGroupId  gid: " << (*git).id() << std::endl;
  }
  // Endcap loop
  for (git = grp->endcapItems().begin(); git != grp->endcapItems().end(); ++git) {
    std::cout << "XtalGroupId  gid: " << (*git).id() << std::endl;
  }

  // Gain Ratios
  edm::ESHandle<EcalChannelStatus> pStatus;
  context.get<EcalChannelStatusRcd>().get(pStatus);
  const EcalChannelStatusMap* ch = pStatus.product();
  EcalChannelStatusMap::const_iterator chit;
  // Barrel loop
  for (chit = ch->barrelItems().begin(); chit != ch->barrelItems().end(); ++chit) {
    EcalChannelStatusCode chst;
    chst = (*chit);
    std::cout << "Ecal channel status  " << chst.getStatusCode() << std::endl;
  }
  // Endcap loop
  for (chit = ch->endcapItems().begin(); chit != ch->endcapItems().end(); ++chit) {
    EcalChannelStatusCode chst;
    chst = (*chit);
    std::cout << "Ecal channel status  " << chst.getStatusCode() << std::endl;
  }

  edm::ESHandle<EcalGainRatios> pRatio;
  context.get<EcalGainRatiosRcd>().get(pRatio);
  const EcalGainRatios* gr = pRatio.product();
  EcalGainRatioMap::const_iterator grit;
  // Barrel loop
  for (grit = gr->barrelItems().begin(); grit != gr->barrelItems().end(); ++grit) {
    EcalMGPAGainRatio mgpa;
    mgpa = (*grit);
    std::cout << "EcalMGPAGainRatio: gain 12/6:  " << mgpa.gain12Over6() << " gain 6/1: " << mgpa.gain6Over1()
              << std::endl;
  }
  // Endcap loop
  for (grit = gr->endcapItems().begin(); grit != gr->endcapItems().end(); ++grit) {
    EcalMGPAGainRatio mgpa;
    mgpa = (*grit);
    std::cout << "EcalMGPAGainRatio: gain 12/6:  " << mgpa.gain12Over6() << " gain 6/1: " << mgpa.gain6Over1()
              << std::endl;
  }

  // Intercalib constants
  edm::ESHandle<EcalIntercalibConstants> pIcal;
  context.get<EcalIntercalibConstantsRcd>().get(pIcal);
  const EcalIntercalibConstants* ical = pIcal.product();
  EcalIntercalibConstantMap::const_iterator icalit;
  // Barrel loop
  for (icalit = ical->barrelItems().begin(); icalit != ical->barrelItems().end(); ++icalit) {
    std::cout << "EcalIntercalibConstant:  icalconst: " << (*icalit) << std::endl;
  }
  // Endcap loop
  for (icalit = ical->endcapItems().begin(); icalit != ical->endcapItems().end(); ++icalit) {
    std::cout << "EcalIntercalibConstant:  icalconst: " << (*icalit) << std::endl;
  }

  edm::ESHandle<EcalIntercalibErrors> pIcalError;
  context.get<EcalIntercalibErrorsRcd>().get(pIcalError);
  const EcalIntercalibErrors* icalerr = pIcalError.product();
  EcalIntercalibErrorMap::const_iterator icalerrit;
  // Barrel loop
  for (icalerrit = icalerr->barrelItems().begin(); icalerrit != icalerr->barrelItems().end(); ++icalerrit) {
    std::cout << "EcalIntercalibConstant:  error: " << (*icalerrit) << std::endl;
  }
  // Endcap loop
  for (icalerrit = icalerr->endcapItems().begin(); icalerrit != icalerr->endcapItems().end(); ++icalerrit) {
    std::cout << "EcalIntercalibConstant:  error: " << (*icalerrit) << std::endl;
  }

  // Time calibration constants
  {
    edm::ESHandle<EcalTimeCalibConstants> pIcal;
    context.get<EcalTimeCalibConstantsRcd>().get(pIcal);
    const EcalTimeCalibConstants* ical = pIcal.product();
    EcalTimeCalibConstantMap::const_iterator icalit;
    // Barrel loop
    for (icalit = ical->barrelItems().begin(); icalit != ical->barrelItems().end(); ++icalit) {
      std::cout << "EcalTimeCalibConstant:  icalconst: " << (*icalit) << std::endl;
    }
    // Endcap loop
    for (icalit = ical->endcapItems().begin(); icalit != ical->endcapItems().end(); ++icalit) {
      std::cout << "EcalTimeCalibConstant:  icalconst: " << (*icalit) << std::endl;
    }

    edm::ESHandle<EcalTimeCalibErrors> pIcalError;
    context.get<EcalTimeCalibErrorsRcd>().get(pIcalError);
    const EcalTimeCalibErrors* icalerr = pIcalError.product();
    EcalTimeCalibErrorMap::const_iterator icalerrit;
    // Barrel loop
    for (icalerrit = icalerr->barrelItems().begin(); icalerrit != icalerr->barrelItems().end(); ++icalerrit) {
      std::cout << "EcalTimeCalibConstant:  error: " << (*icalerrit) << std::endl;
    }
    // Endcap loop
    for (icalerrit = icalerr->endcapItems().begin(); icalerrit != icalerr->endcapItems().end(); ++icalerrit) {
      std::cout << "EcalTimeCalibConstant:  error: " << (*icalerrit) << std::endl;
    }
  }

  // fetch TB weights
  edm::ESHandle<EcalTBWeights> pWgts;
  context.get<EcalTBWeightsRcd>().get(pWgts);
  const EcalTBWeights* wgts = pWgts.product();
  std::cout << "EcalTBWeightMap.size(): " << wgts->getMap().size() << std::endl;

  //   // look up the correct weights for this  xtal
  //   //EcalXtalGroupId gid( git->second );
  //   EcalTBWeights::EcalTDCId tdcid(1);
  for (EcalTBWeights::EcalTBWeightMap::const_iterator wit = wgts->getMap().begin(); wit != wgts->getMap().end();
       ++wit) {
    std::cout << "EcalWeights " << wit->first.first.id() << "," << wit->first.second << std::endl;
    wit->second.print(std::cout);
    std::cout << std::endl;
  }

  // get from offline DB the last valid laser set
  edm::ESHandle<EcalLaserAPDPNRatios> apdPnRatiosHandle;
  context.get<EcalLaserAPDPNRatiosRcd>().get(apdPnRatiosHandle);

  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap& laserRatiosMap = apdPnRatiosHandle.product()->getLaserMap();
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap& laserTimeMap = apdPnRatiosHandle.product()->getTimeMap();

  EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap::const_iterator apdPnRatiosit;
  // Barrel loop
  for (apdPnRatiosit = laserRatiosMap.barrelItems().begin(); apdPnRatiosit != laserRatiosMap.barrelItems().end();
       ++apdPnRatiosit) {
    EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdPnRatioPair = (*apdPnRatiosit);
    std::cout << "EcalAPDPnRatio: first " << apdPnRatioPair.p1 << " second " << apdPnRatioPair.p2 << std::endl;
  }
  // Endcap loop
  for (apdPnRatiosit = laserRatiosMap.endcapItems().begin(); apdPnRatiosit != laserRatiosMap.endcapItems().end();
       ++apdPnRatiosit) {
    EcalLaserAPDPNRatios::EcalLaserAPDPNpair apdPnRatioPair = (*apdPnRatiosit);
    std::cout << "EcalAPDPnRatio: first " << apdPnRatioPair.p1 << " second " << apdPnRatioPair.p2 << std::endl;
  }
  //TimeStampLoop
  for (unsigned int i = 0; i < laserTimeMap.size(); ++i) {
    EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp = laserTimeMap[i];
    std::cout << "EcalAPDPnRatio: timestamp : " << i << " " << timestamp.t1.value() << " , " << timestamp.t2.value()
              << endl;
  }

  // get from offline DB the last valid laser set
  edm::ESHandle<EcalLaserAlphas> alphasHandle;
  context.get<EcalLaserAlphasRcd>().get(alphasHandle);
  const EcalLaserAlphaMap* alphaMap = alphasHandle.product();
  EcalLaserAlphaMap::const_iterator alphait;
  // Barrel loop
  for (alphait = alphaMap->barrelItems().begin(); alphait != alphaMap->barrelItems().end(); ++alphait) {
    std::cout << "EcalLaserAlphas:  icalconst: " << (*alphait) << std::endl;
  }
  // Endcap loop
  for (alphait = alphaMap->endcapItems().begin(); alphait != alphaMap->endcapItems().end(); ++alphait) {
    std::cout << "EcalLaserAlphas:  icalconst: " << (*alphait) << std::endl;
  }

  // get from offline DB the last valid laser set
  edm::ESHandle<EcalLaserAPDPNRatiosRef> apdPnRatioRefHandle;
  context.get<EcalLaserAPDPNRatiosRefRcd>().get(apdPnRatioRefHandle);
  const EcalLaserAPDPNRatiosRefMap* apdPnRatioRefMap = apdPnRatioRefHandle.product();
  EcalLaserAPDPNRatiosRefMap::const_iterator apdPnRatioRefIt;
  // Barrel loop
  for (apdPnRatioRefIt = apdPnRatioRefMap->barrelItems().begin();
       apdPnRatioRefIt != apdPnRatioRefMap->barrelItems().end();
       ++apdPnRatioRefIt) {
    std::cout << "EcalLaserAPDPNRatiosRef:  icalconst: " << (*apdPnRatioRefIt) << std::endl;
  }
  // Endcap loop
  for (apdPnRatioRefIt = apdPnRatioRefMap->endcapItems().begin();
       apdPnRatioRefIt != apdPnRatioRefMap->endcapItems().end();
       ++apdPnRatioRefIt) {
    std::cout << "EcalLaserAPDPNRatiosRef:  icalconst: " << (*apdPnRatioRefIt) << std::endl;
  }

  edm::ESHandle<EcalMappingElectronics> ecalmapping;
  context.get<EcalMappingElectronicsRcd>().get(ecalmapping);
  const EcalMappingElectronics* Mapping = ecalmapping.product();
  const std::vector<EcalMappingElement>& ee = Mapping->endcapItems();
  for (size_t iMap = 0; iMap < ee.size(); iMap++) {
    std::cout << "EcalMappingElectronics: " << ee[iMap].electronicsid << " " << ee[iMap].triggerid << std::endl;
  }

}  //end of ::Analyze()
DEFINE_FWK_MODULE(EcalObjectAnalyzer);
