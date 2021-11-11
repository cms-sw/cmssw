
/*----------------------------------------------------------------------

Sh. Rahatlou, University of Rome & INFN
simple analyzer to dump information about ECAL cond objects

----------------------------------------------------------------------*/

#include <sstream>
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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

class EcalObjectAnalyzer : public edm::global::EDAnalyzer<> {
public:
  explicit EcalObjectAnalyzer(edm::ParameterSet const &p);
  ~EcalObjectAnalyzer() override = default;

  void analyze(edm::StreamID, edm::Event const &, edm::EventSetup const &) const override;

private:
  const edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedestalsToken_;
  const edm::ESGetToken<EcalADCToGeVConstant, EcalADCToGeVConstantRcd> adcToGeVConstantToken_;
  const edm::ESGetToken<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd> weightXtalGroupsToken_;
  const edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> channelStatusToken_;
  const edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> gainRatiosToken_;
  const edm::ESGetToken<EcalIntercalibConstants, EcalIntercalibConstantsRcd> intercalibConstantsToken_;
  const edm::ESGetToken<EcalIntercalibErrors, EcalIntercalibErrorsRcd> intercalibErrorsToken_;
  const edm::ESGetToken<EcalTimeCalibConstants, EcalTimeCalibConstantsRcd> timeCalibConstantsToken_;
  const edm::ESGetToken<EcalTimeCalibErrors, EcalTimeCalibErrorsRcd> timeCalibErrorsToken_;
  const edm::ESGetToken<EcalTBWeights, EcalTBWeightsRcd> tbWeightsToken_;
  const edm::ESGetToken<EcalLaserAPDPNRatios, EcalLaserAPDPNRatiosRcd> laserAPDPNRatiosToken_;
  const edm::ESGetToken<EcalLaserAlphas, EcalLaserAlphasRcd> laserAlphasToken_;
  const edm::ESGetToken<EcalLaserAPDPNRatiosRef, EcalLaserAPDPNRatiosRefRcd> laserAPDPNRatiosRefToken_;
  const edm::ESGetToken<EcalMappingElectronics, EcalMappingElectronicsRcd> mappingElectronicsToken_;
};

EcalObjectAnalyzer::EcalObjectAnalyzer(edm::ParameterSet const &p)
    : pedestalsToken_(esConsumes()),
      adcToGeVConstantToken_(esConsumes()),
      weightXtalGroupsToken_(esConsumes()),
      channelStatusToken_(esConsumes()),
      gainRatiosToken_(esConsumes()),
      intercalibConstantsToken_(esConsumes()),
      intercalibErrorsToken_(esConsumes()),
      timeCalibConstantsToken_(esConsumes()),
      timeCalibErrorsToken_(esConsumes()),
      tbWeightsToken_(esConsumes()),
      laserAPDPNRatiosToken_(esConsumes()),
      laserAlphasToken_(esConsumes()),
      laserAPDPNRatiosRefToken_(esConsumes()),
      mappingElectronicsToken_(esConsumes()) {}

void EcalObjectAnalyzer::analyze(edm::StreamID, const edm::Event &e, const edm::EventSetup &context) const {
  using namespace edm::eventsetup;
  // Context is not used.
  edm::LogVerbatim("EcalObjectAnalyzer") << ">>> EcalObjectAnalyzer: processing run " << e.id().run()
                                         << " event: " << e.id().event() << "\n";

  // ADC -> GeV Scale
  const auto &agc = context.getData(adcToGeVConstantToken_);
  edm::LogVerbatim("EcalObjectAnalyzer") << "Global ADC->GeV scale: EB " << agc.getEBValue() << " GeV/ADC count"
                                         << " EE " << agc.getEEValue() << " GeV/ADC count\n";

  const auto &myped = context.getData(pedestalsToken_);
  // Barrel loop
  int cnt = 0;
  for (const auto &item : myped.barrelItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer")
        << "EcalPedestal: "
        << " BARREL " << cnt << " "
        << "  mean_x1:  " << item.mean_x1 << " rms_x1: " << item.rms_x1 << "  mean_x6:  " << item.mean_x6
        << " rms_x6: " << item.rms_x6 << "  mean_x12: " << item.mean_x12 << " rms_x12: " << item.rms_x12 << "\n";
    ++cnt;
  }
  // Endcap loop
  for (const auto &item : myped.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer")
        << "EcalPedestal: "
        << " ENDCAP "
        << "  mean_x1:  " << item.mean_x1 << " rms_x1: " << item.rms_x1 << "  mean_x6:  " << item.mean_x6
        << " rms_x6: " << item.rms_x6 << "  mean_x12: " << item.mean_x12 << " rms_x12: " << item.rms_x12 << "\n";
  }

  // fetch map of groups of xtals
  const auto &grp = context.getData(weightXtalGroupsToken_);
  // Barrel loop
  for (const auto &item : grp.barrelItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "XtalGroupId  gid: " << item.id() << "\n";
  }
  // Endcap loop
  for (const auto &item : grp.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "XtalGroupId  gid: " << item.id() << "\n";
  }

  // Gain Ratios
  const auto &ch = context.getData(channelStatusToken_);
  // Barrel loop
  for (const auto &chst : ch.barrelItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "Ecal channel status  " << chst.getStatusCode() << "\n";
  }
  // Endcap loop
  for (const auto &chst : ch.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "Ecal channel status  " << chst.getStatusCode() << "\n";
  }

  const auto &gr = context.getData(gainRatiosToken_);
  // Barrel loop
  for (const auto &mgpa : gr.barrelItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer")
        << "EcalMGPAGainRatio: gain 12/6:  " << mgpa.gain12Over6() << " gain 6/1: " << mgpa.gain6Over1() << "\n";
  }
  // Endcap loop
  for (const auto &mgpa : gr.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer")
        << "EcalMGPAGainRatio: gain 12/6:  " << mgpa.gain12Over6() << " gain 6/1: " << mgpa.gain6Over1() << "\n";
  }

  // Intercalib constants
  const auto &ical = context.getData(intercalibConstantsToken_);
  // Barrel loop
  for (const auto &item : ical.barrelItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "EcalIntercalibConstant:  icalconst: " << item << "\n";
  }
  // Endcap loop
  for (const auto &item : ical.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "EcalIntercalibConstant:  icalconst: " << item << "\n";
  }

  const auto &icalerr = context.getData(intercalibErrorsToken_);
  // Barrel loop
  for (const auto &item : icalerr.barrelItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "EcalIntercalibConstant:  error: " << item << "\n";
  }
  // Endcap loop
  for (const auto &item : icalerr.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "EcalIntercalibConstant:  error: " << item << "\n";
  }

  // Time calibration constants
  {
    const auto &ical = context.getData(timeCalibConstantsToken_);
    // Barrel loop
    for (const auto &item : ical.barrelItems()) {
      edm::LogVerbatim("EcalObjectAnalyzer") << "EcalTimeCalibConstant:  icalconst: " << item << "\n";
    }
    // Endcap loop
    for (const auto &item : ical.endcapItems()) {
      edm::LogVerbatim("EcalObjectAnalyzer") << "EcalTimeCalibConstant:  icalconst: " << item << "\n";
    }

    const auto &icalerr = context.getData(timeCalibErrorsToken_);
    // Barrel loop
    for (const auto &item : icalerr.barrelItems()) {
      edm::LogVerbatim("EcalObjectAnalyzer") << "EcalTimeCalibConstant:  error: " << item << "\n";
    }
    // Endcap loop
    for (const auto &item : icalerr.endcapItems()) {
      edm::LogVerbatim("EcalObjectAnalyzer") << "EcalTimeCalibConstant:  error: " << item << "\n";
    }
  }

  // fetch TB weights
  const auto &wgts = context.getData(tbWeightsToken_);
  edm::LogVerbatim("EcalObjectAnalyzer") << "EcalTBWeightMap.size(): " << wgts.getMap().size() << "\n";

  //   // look up the correct weights for this  xtal
  //   //EcalXtalGroupId gid( git->second );
  //   EcalTBWeights::EcalTDCId tdcid(1);
  for (const auto &item : wgts.getMap()) {
    edm::LogVerbatim("EcalObjectAnalyzer")
        << "EcalWeights " << item.first.first.id() << "," << item.first.second << "\n";
    std::ostringstream oss;
    item.second.print(oss);
    edm::LogVerbatim("EcalObjectAnalyzer") << oss.str() << "\n";
  }

  // get from offline DB the last valid laser set
  const auto &apdPnRatios = context.getData(laserAPDPNRatiosToken_);

  const EcalLaserAPDPNRatios::EcalLaserAPDPNRatiosMap &laserRatiosMap = apdPnRatios.getLaserMap();
  const EcalLaserAPDPNRatios::EcalLaserTimeStampMap &laserTimeMap = apdPnRatios.getTimeMap();

  // Barrel loop
  for (const auto &apdPnRatioPair : laserRatiosMap.barrelItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer")
        << "EcalAPDPnRatio: first " << apdPnRatioPair.p1 << " second " << apdPnRatioPair.p2 << "\n";
  }
  // Endcap loop
  for (const auto &apdPnRatioPair : laserRatiosMap.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer")
        << "EcalAPDPnRatio: first " << apdPnRatioPair.p1 << " second " << apdPnRatioPair.p2 << "\n";
  }
  //TimeStampLoop
  for (unsigned int i = 0; i < laserTimeMap.size(); ++i) {
    EcalLaserAPDPNRatios::EcalLaserTimeStamp timestamp = laserTimeMap[i];
    edm::LogVerbatim("EcalObjectAnalyzer")
        << "EcalAPDPnRatio: timestamp : " << i << " " << timestamp.t1.value() << " , " << timestamp.t2.value() << endl;
  }

  // get from offline DB the last valid laser set
  const auto &alphaMap = context.getData(laserAlphasToken_);
  // Barrel loop
  for (const auto &item : alphaMap.barrelItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "EcalLaserAlphas:  icalconst: " << item << "\n";
  }
  // Endcap loop
  for (const auto &item : alphaMap.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "EcalLaserAlphas:  icalconst: " << item << "\n";
  }

  // get from offline DB the last valid laser set
  const auto &apdPnRatioRefMap = context.getData(laserAPDPNRatiosRefToken_);
  // Barrel loop
  for (const auto &item : apdPnRatioRefMap.barrelItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "EcalLaserAPDPNRatiosRef:  icalconst: " << item << "\n";
  }
  // Endcap loop
  for (const auto &item : apdPnRatioRefMap.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer") << "EcalLaserAPDPNRatiosRef:  icalconst: " << item << "\n";
  }

  const auto &mapping = context.getData(mappingElectronicsToken_);
  for (const auto &item : mapping.endcapItems()) {
    edm::LogVerbatim("EcalObjectAnalyzer")
        << "EcalMappingElectronics: " << item.electronicsid << " " << item.triggerid << "\n";
  }

}  //end of ::Analyze()
DEFINE_FWK_MODULE(EcalObjectAnalyzer);
