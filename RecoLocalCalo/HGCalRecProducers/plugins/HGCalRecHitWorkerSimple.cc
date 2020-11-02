#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitWorkerSimple.h"

#include <memory>

#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

HGCalRecHitWorkerSimple::HGCalRecHitWorkerSimple(const edm::ParameterSet& ps) : HGCalRecHitWorkerBaseClass(ps) {
  rechitMaker_ = std::make_unique<HGCalRecHitSimpleAlgo>();
  tools_ = std::make_unique<hgcal::RecHitTools>();
  constexpr float keV2GeV = 1e-6;
  // HGCee constants
  hgcEE_keV2DIGI_ = ps.getParameter<double>("HGCEE_keV2DIGI");
  hgcEE_fCPerMIP_ = ps.getParameter<std::vector<double> >("HGCEE_fCPerMIP");
  hgcEE_isSiFE_ = ps.getParameter<bool>("HGCEE_isSiFE");
  hgceeUncalib2GeV_ = keV2GeV / hgcEE_keV2DIGI_;

  // HGChef constants
  hgcHEF_keV2DIGI_ = ps.getParameter<double>("HGCHEF_keV2DIGI");
  hgcHEF_fCPerMIP_ = ps.getParameter<std::vector<double> >("HGCHEF_fCPerMIP");
  hgcHEF_isSiFE_ = ps.getParameter<bool>("HGCHEF_isSiFE");
  hgchefUncalib2GeV_ = keV2GeV / hgcHEF_keV2DIGI_;

  // HGCheb constants
  hgcHEB_keV2DIGI_ = ps.getParameter<double>("HGCHEB_keV2DIGI");
  hgcHEB_isSiFE_ = ps.getParameter<bool>("HGCHEB_isSiFE");
  hgchebUncalib2GeV_ = keV2GeV / hgcHEB_keV2DIGI_;

  // HGChfnose constants
  hgcHFNose_keV2DIGI_ = ps.getParameter<double>("HGCHFNose_keV2DIGI");
  hgcHFNose_fCPerMIP_ = ps.getParameter<std::vector<double> >("HGCHFNose_fCPerMIP");
  hgcHFNose_isSiFE_ = ps.getParameter<bool>("HGCHFNose_isSiFE");
  hgchfnoseUncalib2GeV_ = keV2GeV / hgcHFNose_keV2DIGI_;

  // layer weights (from Valeri/Arabella)
  const auto& dweights = ps.getParameter<std::vector<double> >("layerWeights");
  for (auto weight : dweights) {
    weights_.push_back(weight);
  }
  const auto& weightnose = ps.getParameter<std::vector<double> >("layerNoseWeights");
  for (auto const& weight : weightnose)
    weightsNose_.emplace_back(weight);

  rechitMaker_->setLayerWeights(weights_);
  rechitMaker_->setNoseLayerWeights(weightsNose_);

  // residual correction for cell thickness
  // first for silicon
  const auto& rcorr = ps.getParameter<std::vector<double> >("thicknessCorrection");
  rcorr_.clear();
  rcorr_.push_back(1.f);
  for (auto corr : rcorr) {
    rcorr_.push_back(1.0 / corr);
  }
  // here for scintillator
  rcorrscint_ = 1.0 / ps.getParameter<double>("sciThicknessCorrection");

  //This is for the index position in CE_H silicon thickness cases
  deltasi_index_regemfac_ = ps.getParameter<int>("deltasi_index_regemfac");
  const auto& rcorrnose = ps.getParameter<std::vector<double> >("thicknessNoseCorrection");
  rcorrNose_.clear();
  rcorrNose_.push_back(1.f);
  for (auto corr : rcorrnose) {
    rcorrNose_.push_back(1.0 / corr);
  }

  hgcEE_noise_fC_ = ps.getParameter<edm::ParameterSet>("HGCEE_noise_fC").getParameter<std::vector<double> >("values");
  hgcEE_cce_ = ps.getParameter<edm::ParameterSet>("HGCEE_cce").getParameter<std::vector<double> >("values");
  hgcHEF_noise_fC_ = ps.getParameter<edm::ParameterSet>("HGCHEF_noise_fC").getParameter<std::vector<double> >("values");
  hgcHEF_cce_ = ps.getParameter<edm::ParameterSet>("HGCHEF_cce").getParameter<std::vector<double> >("values");
  hgcHEB_noise_MIP_ = ps.getParameter<edm::ParameterSet>("HGCHEB_noise_MIP").getParameter<double>("noise_MIP");
  hgcHFNose_noise_fC_ =
      ps.getParameter<edm::ParameterSet>("HGCHFNose_noise_fC").getParameter<std::vector<double> >("values");
  hgcHFNose_cce_ = ps.getParameter<edm::ParameterSet>("HGCHFNose_cce").getParameter<std::vector<double> >("values");

  // don't produce rechit if detid is a ghost one
  rangeMatch_ = ps.getParameter<uint32_t>("rangeMatch");
  rangeMask_ = ps.getParameter<uint32_t>("rangeMask");

  // error for recHit time
  timeEstimatorSi_ = hgcalsimclustertime::ComputeClusterTime(ps.getParameter<double>("minValSiPar"),
                                                             ps.getParameter<double>("maxValSiPar"),
                                                             ps.getParameter<double>("constSiPar"),
                                                             ps.getParameter<double>("noiseSiPar"));
}

void HGCalRecHitWorkerSimple::set(const edm::EventSetup& es) {
  edm::ESHandle<CaloGeometry> geom;
  es.get<CaloGeometryRecord>().get(geom);
  tools_->setGeometry(*geom);
  rechitMaker_->set(*geom);
  if (hgcEE_isSiFE_) {
    edm::ESHandle<HGCalGeometry> hgceeGeoHandle;
    es.get<IdealGeometryRecord>().get("HGCalEESensitive", hgceeGeoHandle);
    ddds_[0] = &(hgceeGeoHandle->topology().dddConstants());
  } else {
    ddds_[0] = nullptr;
  }
  if (hgcHEF_isSiFE_) {
    edm::ESHandle<HGCalGeometry> hgchefGeoHandle;
    es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive", hgchefGeoHandle);
    ddds_[1] = &(hgchefGeoHandle->topology().dddConstants());
  } else {
    ddds_[1] = nullptr;
  }
  ddds_[2] = nullptr;
  if (hgcHFNose_isSiFE_) {
    edm::ESHandle<HGCalGeometry> hgchfnoseGeoHandle;
    es.get<IdealGeometryRecord>().get("HGCalHFNoseSensitive", hgchfnoseGeoHandle);
    ddds_[3] = &(hgchfnoseGeoHandle->topology().dddConstants());
  } else {
    ddds_[3] = nullptr;
  }
}

bool HGCalRecHitWorkerSimple::run(const edm::Event& evt,
                                  const HGCUncalibratedRecHit& uncalibRH,
                                  HGCRecHitCollection& result) {
  DetId detid = uncalibRH.id();
  // don't produce rechit if detid is a ghost one

  if (detid.det() == DetId::Forward || detid.det() == DetId::Hcal) {
    if ((detid & rangeMask_) == rangeMatch_)
      return false;
  }

  int thickness = -1;
  float sigmaNoiseGeV = 0.f;
  unsigned int layer = tools_->getLayerWithOffset(detid);
  float cce_correction = 1.0;
  int idtype(0);

  switch (detid.det()) {
    case DetId::HGCalEE:
      idtype = hgcee;
      thickness = 1 + HGCSiliconDetId(detid).type();
      break;
    case DetId::HGCalHSi:
      idtype = hgcfh;
      thickness = 1 + HGCSiliconDetId(detid).type();
      break;
    case DetId::HGCalHSc:
      idtype = hgcbh;
      break;
    default:
      switch (detid.subdetId()) {
        case HGCEE:
          idtype = hgcee;
          thickness = ddds_[detid.subdetId() - 3]->waferTypeL(HGCalDetId(detid).wafer());
          break;
        case HGCHEF:
          idtype = hgcfh;
          thickness = ddds_[detid.subdetId() - 3]->waferTypeL(HGCalDetId(detid).wafer());
          break;
        case HcalEndcap:
          [[fallthrough]];
        case HGCHEB:
          idtype = hgcbh;
          break;
        case HFNose:
          idtype = hgchfnose;
          thickness = 1 + HFNoseDetId(detid).type();
          break;
        default:
          break;
      }
  }
  switch (idtype) {
    case hgcee:
      rechitMaker_->setADCToGeVConstant(float(hgceeUncalib2GeV_));
      cce_correction = hgcEE_cce_[thickness - 1];
      sigmaNoiseGeV =
          1e-3 * weights_[layer] * rcorr_[thickness] * hgcEE_noise_fC_[thickness - 1] / hgcEE_fCPerMIP_[thickness - 1];
      break;
    case hgcfh:
      rechitMaker_->setADCToGeVConstant(float(hgchefUncalib2GeV_));
      cce_correction = hgcHEF_cce_[thickness - 1];
      sigmaNoiseGeV = 1e-3 * weights_[layer] * rcorr_[thickness + deltasi_index_regemfac_] *
                      hgcHEF_noise_fC_[thickness - 1] / hgcHEF_fCPerMIP_[thickness - 1];
      break;
    case hgcbh:
      rechitMaker_->setADCToGeVConstant(float(hgchebUncalib2GeV_));
      sigmaNoiseGeV = 1e-3 * hgcHEB_noise_MIP_ * weights_[layer] * rcorrscint_;
      break;
    case hgchfnose:
      rechitMaker_->setADCToGeVConstant(float(hgchfnoseUncalib2GeV_));
      cce_correction = hgcHFNose_cce_[thickness - 1];
      sigmaNoiseGeV = 1e-3 * weightsNose_[layer] * rcorrNose_[thickness] * hgcHFNose_noise_fC_[thickness - 1] /
                      hgcHFNose_fCPerMIP_[thickness - 1];
      break;
    default:
      throw cms::Exception("NonHGCRecHit") << "Rechit with detid = " << detid.rawId() << " is not HGC!";
  }

  // make the rechit and put in the output collection

  HGCRecHit myrechit(rechitMaker_->makeRecHit(uncalibRH, 0));
  double new_E = myrechit.energy();
  if (detid.det() == DetId::Forward && detid.subdetId() == ForwardSubdetector::HFNose) {
    new_E *= (thickness == -1 ? 1.0 : rcorrNose_[thickness]) / cce_correction;
  }  //regional factors for silicon in CE_H
  else if (idtype == hgcfh) {
    new_E *= rcorr_[thickness + deltasi_index_regemfac_] / cce_correction;
  }  //regional factors for scintillator and silicon in CE_E
  else {
    new_E *= (thickness == -1 ? rcorrscint_ : rcorr_[thickness]) / cce_correction;
  }

  myrechit.setEnergy(new_E);
  float SoN = new_E / sigmaNoiseGeV;
  myrechit.setSignalOverSigmaNoise(SoN);

  if (detid.det() == DetId::HGCalHSc || myrechit.time() < 0.) {
    myrechit.setTimeError(-1.);
  } else {
    float timeError = timeEstimatorSi_.getTimeError("recHit", SoN);
    myrechit.setTimeError(timeError);
  }

  result.push_back(myrechit);

  return true;
}

HGCalRecHitWorkerSimple::~HGCalRecHitWorkerSimple() {}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN(HGCalRecHitWorkerFactory, HGCalRecHitWorkerSimple, "HGCalRecHitWorkerSimple");
