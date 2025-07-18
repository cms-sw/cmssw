#include "ZdcHitReconstructor_Run3.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

#include <iostream>

#include <Eigen/Dense>

#include <vector>
namespace zdchelper {
  void setZDCSaturation(ZDCRecHit& rh, QIE10DataFrame& digi, int maxValue) {
    for (int i = 0; i < digi.samples(); i++) {
      if (digi[i].adc() >= maxValue) {
        rh.setFlagField(1, HcalCaloFlagLabels::ADCSaturationBit);
        break;
      }
    }
  }

}  // namespace zdchelper

ZdcHitReconstructor_Run3::ZdcHitReconstructor_Run3(edm::ParameterSet const& conf)

    : reco_(conf.getParameter<int>("recoMethod")),
      saturationFlagSetter_(nullptr),
      det_(DetId::Hcal),
      correctionMethodEM_(conf.getParameter<int>("correctionMethodEM")),
      correctionMethodHAD_(conf.getParameter<int>("correctionMethodHAD")),
      correctionMethodRPD_(conf.getParameter<int>("correctionMethodRPD")),
      ootpuRatioEM_(conf.getParameter<double>("ootpuRatioEM")),
      ootpuRatioHAD_(conf.getParameter<double>("ootpuRatioHAD")),
      ootpuRatioRPD_(conf.getParameter<double>("ootpuRatioRPD")),
      ootpuFracEM_(conf.getParameter<double>("ootpuFracEM")),
      ootpuFracHAD_(conf.getParameter<double>("ootpuFracHAD")),
      ootpuFracRPD_(conf.getParameter<double>("ootpuFracRPD")),
      chargeRatiosEM_(conf.getParameter<std::vector<double>>("chargeRatiosEM")),
      chargeRatiosHAD_(conf.getParameter<std::vector<double>>("chargeRatiosHAD")),
      chargeRatiosRPD_(conf.getParameter<std::vector<double>>("chargeRatiosRPD")),
      bxTs_(conf.getParameter<std::vector<unsigned int>>("bxTs")),
      nTs_(conf.getParameter<int>("nTs")),
      forceSOI_(conf.getParameter<bool>("forceSOI")),
      signalSOI_(conf.getParameter<std::vector<unsigned int>>("signalSOI")),
      noiseSOI_(conf.getParameter<std::vector<unsigned int>>("noiseSOI")),
      setSaturationFlags_(conf.getParameter<bool>("setSaturationFlags")),
      dropZSmarkedPassed_(conf.getParameter<bool>("dropZSmarkedPassed")),
      skipRPD_(conf.getParameter<bool>("skipRPD")) {
  tok_input_QIE10 = consumes<QIE10DigiCollection>(conf.getParameter<edm::InputTag>("digiLabelQIE10ZDC"));

  std::string subd = conf.getParameter<std::string>("Subdetector");

  if (setSaturationFlags_) {
    const edm::ParameterSet& pssat = conf.getParameter<edm::ParameterSet>("saturationParameters");
    maxADCvalue_ = pssat.getParameter<int>("maxADCvalue");
  }
  if (!strcasecmp(subd.c_str(), "ZDC")) {
    det_ = DetId::Calo;
    subdet_ = HcalZDCDetId::SubdetectorId;
    produces<ZDCRecHitCollection>();
  } else if (!strcasecmp(subd.c_str(), "CALIB")) {
    subdet_ = HcalOther;
    subdetOther_ = HcalCalibration;
    produces<HcalCalibRecHitCollection>();
  } else {
    std::cout << "ZdcHitReconstructor_Run3 is not associated with a specific subdetector!" << std::endl;
  }
  reco_.initCorrectionMethod(correctionMethodEM_, 1);
  reco_.initCorrectionMethod(correctionMethodHAD_, 2);
  reco_.initCorrectionMethod(correctionMethodRPD_, 4);
  reco_.initTemplateFit(bxTs_, chargeRatiosEM_, nTs_, 1);
  reco_.initTemplateFit(bxTs_, chargeRatiosHAD_, nTs_, 2);
  reco_.initTemplateFit(bxTs_, chargeRatiosRPD_, nTs_, 4);
  reco_.initRatioSubtraction(ootpuRatioEM_, ootpuFracEM_, 1);
  reco_.initRatioSubtraction(ootpuRatioHAD_, ootpuFracHAD_, 2);
  reco_.initRatioSubtraction(ootpuRatioRPD_, ootpuFracRPD_, 4);
  // ES tokens
  htopoToken_ = esConsumes<HcalTopology, HcalRecNumberingRecord, edm::Transition::BeginRun>();
  paramsToken_ = esConsumes<HcalLongRecoParams, HcalLongRecoParamsRcd, edm::Transition::BeginRun>();
  conditionsToken_ = esConsumes<HcalDbService, HcalDbRecord>();
  qualToken_ = esConsumes<HcalChannelQuality, HcalChannelQualityRcd>(edm::ESInputTag("", "withTopo"));
  sevToken_ = esConsumes<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd>();
}

ZdcHitReconstructor_Run3::~ZdcHitReconstructor_Run3() { delete saturationFlagSetter_; }

void ZdcHitReconstructor_Run3::beginRun(edm::Run const& r, edm::EventSetup const& es) {
  const HcalTopology& htopo = es.getData(htopoToken_);
  const HcalLongRecoParams& p = es.getData(paramsToken_);
  longRecoParams_ = std::make_unique<HcalLongRecoParams>(p);
  longRecoParams_->setTopo(&htopo);
}

void ZdcHitReconstructor_Run3::endRun(edm::Run const& r, edm::EventSetup const& es) {}

void ZdcHitReconstructor_Run3::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  // get conditions
  const HcalDbService* conditions = &eventSetup.getData(conditionsToken_);
  const HcalChannelQuality* myqual = &eventSetup.getData(qualToken_);
  const HcalSeverityLevelComputer* mySeverity = &eventSetup.getData(sevToken_);

  // define vectors to pass noiseTS and signalTS
  std::vector<unsigned int> mySignalTS;
  std::vector<unsigned int> myNoiseTS;

  if (det_ == DetId::Calo && subdet_ == HcalZDCDetId::SubdetectorId) {
    edm::Handle<QIE10DigiCollection> digi;
    e.getByToken(tok_input_QIE10, digi);

    // create empty output
    auto rec = std::make_unique<ZDCRecHitCollection>();
    rec->reserve(digi->size());

    // testing QEI10 conditions
    for (auto it = digi->begin(); it != digi->end(); it++) {
      QIE10DataFrame QIE10_i = static_cast<QIE10DataFrame>(*it);
      HcalZDCDetId cell = QIE10_i.id();
      bool isRPD = cell.section() == 4;
      if (isRPD && skipRPD_)
        continue;
      if (cell.section() == 1 && cell.channel() > 5)
        continue;  // ignore extra EM channels

      DetId detcell = (DetId)cell;

      // check on cells to be ignored and dropped: (rof,20.Feb.09)
      const HcalChannelStatus* mydigistatus = myqual->getValues(detcell.rawId());
      if (mySeverity->dropChannel(mydigistatus->getValue()))
        continue;
      if (dropZSmarkedPassed_)
        if (QIE10_i.zsMarkAndPass())
          continue;

      const HcalCalibrations& calibrations = conditions->getHcalCalibrations(cell);
      const HcalQIECoder* channelCoder = conditions->getHcalCoder(cell);
      const HcalQIEShape* shape = conditions->getHcalShape(channelCoder);
      HcalCoderDb coder(*channelCoder, *shape);

      // pass the effective pedestals to rec hit since both ped value and width used in subtraction of pedestals
      const HcalPedestal* effPeds = conditions->getEffectivePedestal(cell);

      if (forceSOI_)
        rec->push_back(reco_.reconstruct(QIE10_i, noiseSOI_, signalSOI_, coder, calibrations, *effPeds));

      else {
        const HcalLongRecoParam* myParams = longRecoParams_->getValues(detcell);
        mySignalTS.clear();
        myNoiseTS.clear();
        mySignalTS = myParams->signalTS();
        myNoiseTS = myParams->noiseTS();

        rec->push_back(reco_.reconstruct(QIE10_i, myNoiseTS, mySignalTS, coder, calibrations, *effPeds));
      }
      // saturationFlagSetter_ doesn't work with QIE10
      // created new function zdchelper::setZDCSaturation to work with QIE10
      (rec->back()).setFlags(0);
      if (setSaturationFlags_)
        zdchelper::setZDCSaturation(rec->back(), QIE10_i, maxADCvalue_);
    }
    // return result
    e.put(std::move(rec));
  }  // else if (det_==DetId::Calo...)

}  // void HcalHitReconstructor::produce(...)

void ZdcHitReconstructor_Run3::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // zdcreco
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiLabelQIE10ZDC", edm::InputTag("hcalDigis", "ZDC"));
  desc.add<std::string>("Subdetector", "ZDC");
  desc.add<bool>("dropZSmarkedPassed", true);
  desc.add<bool>("skipRPD", true);
  desc.add<int>("recoMethod", 1);
  desc.add<int>("correctionMethodEM", 1);
  desc.add<int>("correctionMethodHAD", 1);
  desc.add<int>("correctionMethodRPD", 0);
  desc.add<double>("ootpuRatioEM", 3.0);
  desc.add<double>("ootpuRatioHAD", 3.0);
  desc.add<double>("ootpuRatioRPD", -1.0);
  desc.add<double>("ootpuFracEM", 1.0);
  desc.add<double>("ootpuFracHAD", 1.0);
  desc.add<double>("ootpuFracRPD", 0.0);
  desc.add<std::vector<double>>("chargeRatiosEM",
                                {
                                    1.0,
                                    0.23157,
                                    0.10477,
                                    0.06312,
                                });
  desc.add<std::vector<double>>("chargeRatiosHAD",
                                {
                                    1.0,
                                    0.23157,
                                    0.10477,
                                    0.06312,
                                });
  desc.add<std::vector<double>>("chargeRatiosRPD",
                                {
                                    1.0,
                                    0.23157,
                                    0.10477,
                                    0.06312,
                                });
  desc.add<std::vector<unsigned int>>("bxTs",
                                      {
                                          0,
                                          2,
                                          4,
                                      });
  desc.add<int>("nTs", 6);
  desc.add<bool>("forceSOI", false);
  desc.add<std::vector<unsigned int>>("signalSOI",
                                      {
                                          2,
                                      });
  desc.add<std::vector<unsigned int>>("noiseSOI",
                                      {
                                          1,
                                      });
  desc.add<bool>("setSaturationFlags", true);
  {
    edm::ParameterSetDescription psd0;
    psd0.add<int>("maxADCvalue", 255);
    desc.add<edm::ParameterSetDescription>("saturationParameters", psd0);
  }
  descriptions.add("zdcrecoRun3", desc);
  // or use the following to generate the label from the module's C++ type
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZdcHitReconstructor_Run3);