
#include "DQM/HcalTasks/interface/ZDCTask.h"
#include <map>

using namespace hcaldqm;
using namespace hcaldqm::constants;
ZDCTask::ZDCTask(edm::ParameterSet const& ps) {
  //	tags
  _tagQIE10 = ps.getUntrackedParameter<edm::InputTag>("tagQIE10", edm::InputTag("hcalDigis"));
  _tokQIE10 = consumes<ZDCDigiCollection>(_tagQIE10);

  //	cuts
  _cut = ps.getUntrackedParameter<double>("cut", 50.0);
  _ped = ps.getUntrackedParameter<int>("ped", 4);
}

/* virtual */ void ZDCTask::bookHistograms(DQMStore::IBooker& ib, edm::Run const& r, edm::EventSetup const& es) {
  //############################## hardcode manually the zdc mapping #############################
  //############################# this follows from https://github.com/cms-sw/cmssw/blob/CMSSW_8_0_X/EventFilter/CastorRawToDigi/src/ZdcUnpacker.cc#L118
  //##############################################################################################
  //////ZDC MAP for NEW data (2015 PbPb are newer)
  //PZDC
  std::map<HcalElectronicsId, DetId> myEMap;
  HcalElectronicsId eid = HcalElectronicsId(0, 1, 0, 3);
  eid.setHTR(18, 8, 1);
  myEMap[eid] = DetId(0x54000051);  //PZDC EM1

  eid = HcalElectronicsId(1, 1, 0, 3);
  eid.setHTR(18, 8, 1);
  myEMap[eid] = DetId(0x54000052);  //PZDC EM2

  eid = HcalElectronicsId(2, 1, 0, 3);
  eid.setHTR(18, 8, 1);
  myEMap[eid] = DetId(0x54000053);  //PZDC EM3

  eid = HcalElectronicsId(0, 2, 0, 3);
  eid.setHTR(18, 8, 1);
  myEMap[eid] = DetId(0x54000061);  //PZDC HAD1

  eid = HcalElectronicsId(1, 2, 0, 3);
  eid.setHTR(18, 8, 1);
  myEMap[eid] = DetId(0x54000054);  //PZDC EM4

  eid = HcalElectronicsId(2, 2, 0, 3);
  eid.setHTR(18, 8, 1);
  myEMap[eid] = DetId(0x54000055);  //PZDC EM5

  eid = HcalElectronicsId(0, 3, 0, 3);
  eid.setHTR(18, 8, 1);
  myEMap[eid] = DetId(0x54000062);  //PZDC HAD2

  eid = HcalElectronicsId(1, 3, 0, 3);
  eid.setHTR(18, 8, 1);
  myEMap[eid] = DetId(0x54000063);  //PZDC HAD3

  eid = HcalElectronicsId(2, 3, 0, 3);
  eid.setHTR(18, 8, 1);
  myEMap[eid] = DetId(0x54000064);  //PZDC HAD4

  //NZDC
  eid = HcalElectronicsId(0, 1, 1, 3);
  eid.setHTR(18, 8, 0);
  myEMap[eid] = DetId(0x54000011);  //NZDC EM1

  eid = HcalElectronicsId(1, 1, 1, 3);
  eid.setHTR(18, 8, 0);
  myEMap[eid] = DetId(0x54000012);  //NZDC EM2

  eid = HcalElectronicsId(2, 1, 1, 3);
  eid.setHTR(18, 8, 0);
  myEMap[eid] = DetId(0x54000013);  //NZDC EM3

  eid = HcalElectronicsId(0, 2, 1, 3);
  eid.setHTR(18, 8, 0);
  myEMap[eid] = DetId(0x54000015);  //NZDC EM5

  eid = HcalElectronicsId(1, 2, 1, 3);
  eid.setHTR(18, 8, 0);
  myEMap[eid] = DetId(0x54000021);  //NZDC HAD1

  eid = HcalElectronicsId(2, 2, 1, 3);
  eid.setHTR(18, 8, 0);
  myEMap[eid] = DetId(0x54000014);  //NZDC EM4

  eid = HcalElectronicsId(0, 3, 1, 3);
  eid.setHTR(18, 8, 0);
  myEMap[eid] = DetId(0x54000022);  //NZDC HAD2

  eid = HcalElectronicsId(1, 3, 1, 3);
  eid.setHTR(18, 8, 0);
  myEMap[eid] = DetId(0x54000023);  //NZDC HAD3

  eid = HcalElectronicsId(2, 3, 1, 3);
  eid.setHTR(18, 8, 0);
  myEMap[eid] = DetId(0x54000024);  //NZDC HAD4
  //##################################### end hardcoding ###################################

  ib.cd();

  //quantities for axis
  hcaldqm::quantity::ValueQuantity xAxisShape(hcaldqm::quantity::fTiming_TS);
  hcaldqm::quantity::ValueQuantity yAxisShape(hcaldqm::quantity::ffC_10000);

  hcaldqm::quantity::ValueQuantity xAxisADC(hcaldqm::quantity::fADC_128);

  //book histos per channel
  for (std::map<HcalElectronicsId, DetId>::const_iterator itr = myEMap.begin(); itr != myEMap.end(); ++itr) {
    char histoname[300];

    sprintf(histoname,
            "%d_%d_%d_%d",
            itr->first.fiberChanId(),
            itr->first.fiberIndex(),
            itr->first.spigot(),
            itr->first.dccid());

    ib.setCurrentFolder("Hcal/ZDCTask/Shape_perChannel");
    _cShape_EChannel[histoname] = ib.bookProfile(histoname,
                                                 histoname,
                                                 xAxisShape.nbins(),
                                                 xAxisShape.min(),
                                                 xAxisShape.max(),
                                                 yAxisShape.nbins(),
                                                 yAxisShape.min(),
                                                 yAxisShape.max());
    _cShape_EChannel[histoname]->setAxisTitle("Timing", 1);
    _cShape_EChannel[histoname]->setAxisTitle("fC QIE8", 2);

    ib.setCurrentFolder("Hcal/ZDCTask/ADC_perChannel");
    _cADC_EChannel[histoname] = ib.book1DD(histoname, histoname, xAxisADC.nbins(), xAxisADC.min(), xAxisADC.max());
    _cADC_EChannel[histoname]->getTH1()->SetBit(
        BIT(hcaldqm::constants::BIT_OFFSET + hcaldqm::quantity::AxisType::fYAxis));
    _cADC_EChannel[histoname]->setAxisTitle("ADC QIE8", 1);

    ib.setCurrentFolder("Hcal/ZDCTask/ADC_vs_TS_perChannel");
    _cADC_vs_TS_EChannel[histoname] = ib.book2D(histoname,
                                                histoname,
                                                xAxisShape.nbins(),
                                                xAxisShape.min(),
                                                xAxisShape.max(),
                                                xAxisADC.nbins(),
                                                xAxisADC.min(),
                                                xAxisADC.max());
    _cADC_vs_TS_EChannel[histoname]->getTH1()->SetBit(
        BIT(hcaldqm::constants::BIT_OFFSET + hcaldqm::quantity::AxisType::fYAxis));
    _cADC_vs_TS_EChannel[histoname]->setAxisTitle("Timing", 1);
    _cADC_vs_TS_EChannel[histoname]->setAxisTitle("ADC QIE8", 2);
  }

  //book global histos
  ib.setCurrentFolder("Hcal/ZDCTask");

  _cShape = ib.bookProfile("Shape",
                           "Shape",
                           xAxisShape.nbins(),
                           xAxisShape.min(),
                           xAxisShape.max(),
                           yAxisShape.nbins(),
                           yAxisShape.min(),
                           yAxisShape.max());
  _cShape->setAxisTitle("Timing", 1);
  _cShape->setAxisTitle("fC QIE8", 2);

  _cADC = ib.book1DD("ADC", "ADC", xAxisADC.nbins(), xAxisADC.min(), xAxisADC.max());
  _cADC->getTH1()->SetBit(BIT(hcaldqm::constants::BIT_OFFSET + hcaldqm::quantity::AxisType::fYAxis));
  _cADC->setAxisTitle("ADC QIE8", 1);

  _cADC_vs_TS = ib.book2D("ADC_vs_TS",
                          "ADC_vs_TS",
                          xAxisShape.nbins(),
                          xAxisShape.min(),
                          xAxisShape.max(),
                          xAxisADC.nbins(),
                          xAxisADC.min(),
                          xAxisADC.max());
  _cADC_vs_TS->getTH1()->SetBit(BIT(hcaldqm::constants::BIT_OFFSET + hcaldqm::quantity::AxisType::fYAxis));
  _cADC_vs_TS->setAxisTitle("Timing", 1);
  _cADC_vs_TS->setAxisTitle("ADC QIE8", 2);
}

/* virtual */ void ZDCTask::analyze(edm::Event const& e, edm::EventSetup const&) {
  edm::Handle<ZDCDigiCollection> cqie10;
  if (!e.getByToken(_tokQIE10, cqie10))
    edm::LogError("Collection ZDCDigiCollection isn't available" + _tagQIE10.label() + " " + _tagQIE10.instance());

  for (uint32_t i = 0; i < cqie10->size(); i++) {
    ZDCDataFrame frame = static_cast<ZDCDataFrame>((*cqie10)[i]);
    HcalElectronicsId eid = frame.elecId();

    char histoname[300];
    sprintf(histoname, "%d_%d_%d_%d", eid.fiberChanId(), eid.fiberIndex(), eid.spigot(), eid.dccid());

    //	compute the signal, ped subracted
    //double q = hcaldqm::utilities::sumQ_v10<ZDCDataFrame>(frame, constants::adc2fC[_ped], 0, frame.size()-1);

    //	iterate thru all TS and fill
    for (int j = 0; j < frame.size(); j++) {
      _cShape_EChannel[histoname]->Fill(j, frame[j].nominal_fC());
      _cShape->Fill(j, frame[j].nominal_fC());

      _cADC_EChannel[histoname]->Fill(frame[j].adc());
      _cADC->Fill(frame[j].adc());

      _cADC_vs_TS_EChannel[histoname]->Fill(j, frame[j].adc());
      _cADC_vs_TS->Fill(j, frame[j].adc());
    }
  }
}

DEFINE_FWK_MODULE(ZDCTask);
