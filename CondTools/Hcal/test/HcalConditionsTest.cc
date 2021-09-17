
/*----------------------------------------------------------------------

R.Ofierzynski - 2.Oct. 2007
   modified to dump all pedestals on screen, see 
   testHcalDBFake.cfg
   testHcalDBFrontier.cfg

----------------------------------------------------------------------*/

#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/HcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalQIETypesRcd.h"
#include "CondFormats/DataRecord/interface/HcalPedestalWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainsRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"
#include "CondFormats/DataRecord/interface/HcalGainWidthsRcd.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/DataRecord/interface/HcalZSThresholdsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"

namespace edmtest {
  class HcalConditionsTest : public edm::EDAnalyzer {
  public:
    explicit HcalConditionsTest(edm::ParameterSet const& p) {
      front = p.getUntrackedParameter<std::string>("outFilePrefix", "Dump");
      tok_ = esConsumes<HcalDbService, HcalDbRecord>();
      tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
    }

    explicit HcalConditionsTest(int i) {
      tok_ = esConsumes<HcalDbService, HcalDbRecord>();
      tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
    }
    virtual ~HcalConditionsTest() {}
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);

    template <class S, class SRcd>
    void dumpIt(S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name);

  private:
    std::string front;
    edm::ESGetToken<HcalDbService, HcalDbRecord> tok_;
    edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
  };

  template <class S, class SRcd>
  void HcalConditionsTest::dumpIt(
      S* myS, SRcd* mySRcd, const edm::Event& e, const edm::EventSetup& context, std::string name) {
    edm::ESGetToken<S, SRcd> tok = esConsumes<S, SRcd>();
    int myrun = e.id().run();
    const S* myobject = &context.getData(tok);

    std::ostringstream file;
    file << front << name.c_str() << "_Run" << myrun << ".txt";
    std::ofstream outStream(file.str().c_str());
    std::cout << "HcalConditionsTest: ---- Dumping " << name.c_str() << " ----" << std::endl;
    HcalDbASCIIIO::dumpObject(outStream, (*myobject));

    if (context.get<HcalPedestalsRcd>().validityInterval().first() == edm::IOVSyncValue::invalidIOVSyncValue())
      std::cout << "error: invalid IOV sync value !" << std::endl;
  }

  void HcalConditionsTest::analyze(const edm::Event& e, const edm::EventSetup& context) {
    using namespace edm::eventsetup;
    std::cout << "HcalConditionsTest::analyze-> I AM IN RUN NUMBER " << e.id().run() << std::endl;
    const HcalTopology* topo = &context.getData(tok_htopo_);

    dumpIt(new HcalElectronicsMap, new HcalElectronicsMapRcd, e, context, "ElectronicsMap");
    dumpIt(new HcalQIEData, new HcalQIEDataRcd, e, context, "QIEData");
    dumpIt(new HcalQIETypes, new HcalQIETypesRcd, e, context, "QIETypes");
    dumpIt(new HcalPedestals(topo, false), new HcalPedestalsRcd, e, context, "Pedestals");
    dumpIt(new HcalPedestalWidths(topo, false), new HcalPedestalWidthsRcd, e, context, "PedestalWidths");
    dumpIt(new HcalGains, new HcalGainsRcd, e, context, "Gains");
    dumpIt(new HcalGainWidths, new HcalGainWidthsRcd, e, context, "GainWidths");
    dumpIt(new HcalRespCorrs, new HcalRespCorrsRcd, e, context, "RespCorrs");
    dumpIt(new HcalChannelQuality, new HcalChannelQualityRcd, e, context, "ChannelQuality");
    dumpIt(new HcalZSThresholds, new HcalZSThresholdsRcd, e, context, "ZSThresholds");

    // get conditions
    const auto& conditions = &context.getData(tok_);
    int cell = HcalDetId(HcalBarrel, -1, 4, 1).rawId();
    const HcalCalibrations& calibrations = conditions->getHcalCalibrations(cell);
    std::cout << HcalDetId(cell) << " RespCorr " << calibrations.respcorr() << " TimeCorr " << calibrations.timecorr()
              << std::endl;
  }

  DEFINE_FWK_MODULE(HcalConditionsTest);
}  // namespace edmtest
