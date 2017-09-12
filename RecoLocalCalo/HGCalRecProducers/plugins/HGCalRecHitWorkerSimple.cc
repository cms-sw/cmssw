#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitWorkerSimple.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

HGCalRecHitWorkerSimple::HGCalRecHitWorkerSimple(const edm::ParameterSet&ps) :
        HGCalRecHitWorkerBaseClass(ps)
{
    rechitMaker_.reset(new HGCalRecHitSimpleAlgo());
    tools_.reset(new hgcal::RecHitTools());
    constexpr float keV2GeV = 1e-6;
    // HGCee constants
    hgcEE_keV2DIGI_ = ps.getParameter<double>("HGCEE_keV2DIGI");
    hgcEE_fCPerMIP_ = ps.getParameter < std::vector<double> > ("HGCEE_fCPerMIP");
    hgcEE_isSiFE_ = ps.getParameter<bool>("HGCEE_isSiFE");
    hgceeUncalib2GeV_ = keV2GeV / hgcEE_keV2DIGI_;

    // HGChef constants
    hgcHEF_keV2DIGI_ = ps.getParameter<double>("HGCHEF_keV2DIGI");
    hgcHEF_fCPerMIP_ = ps.getParameter < std::vector<double> > ("HGCHEF_fCPerMIP");
    hgcHEF_isSiFE_ = ps.getParameter<bool>("HGCHEF_isSiFE");
    hgchefUncalib2GeV_ = keV2GeV / hgcHEF_keV2DIGI_;

    // HGCheb constants
    hgcHEB_keV2DIGI_ = ps.getParameter<double>("HGCHEB_keV2DIGI");
    hgcHEB_isSiFE_ = ps.getParameter<bool>("HGCHEB_isSiFE");
    hgchebUncalib2GeV_ = keV2GeV / hgcHEB_keV2DIGI_;

    // layer weights (from Valeri/Arabella)
    const auto& dweights = ps.getParameter < std::vector<double> > ("layerWeights");
    for (auto weight : dweights)
    {
        weights_.push_back(weight);
    }

    rechitMaker_->setLayerWeights(weights_);

    // residual correction for cell thickness
    const auto& rcorr = ps.getParameter < std::vector<double> > ("thicknessCorrection");
    rcorr_.clear();
    rcorr_.push_back(1.f);
    for (auto corr : rcorr)
    {
        rcorr_.push_back(1.0 / corr);
    }


    hgcEE_noise_fC_ = ps.getParameter < std::vector<double> > ("HGCEE_noise_fC");
    hgcEE_cce_ = ps.getParameter< std::vector<double> > ("HGCEE_cce");
    hgcHEF_noise_fC_ = ps.getParameter < std::vector<double> > ("HGCHEF_noise_fC");
    hgcHEF_cce_ = ps.getParameter< std::vector<double> > ("HGCHEF_cce");
    hgcHEB_noise_MIP_ = ps.getParameter<double>("HGCHEB_noise_MIP");

    // don't produce rechit if detid is a ghost one
    rangeMatch_ = ps.getParameter<uint32_t>("rangeMatch");
    rangeMask_  = ps.getParameter<uint32_t>("rangeMask");
}

void HGCalRecHitWorkerSimple::set(const edm::EventSetup& es)
{
    tools_->getEventSetup(es);
    if (hgcEE_isSiFE_)
    {
        edm::ESHandle < HGCalGeometry > hgceeGeoHandle;
        es.get<IdealGeometryRecord>().get("HGCalEESensitive", hgceeGeoHandle);
        ddds_[0] = &(hgceeGeoHandle->topology().dddConstants());
    }
    else
    {
        ddds_[0] = nullptr;
    }
    if (hgcHEF_isSiFE_)
    {
        edm::ESHandle < HGCalGeometry > hgchefGeoHandle;
        es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive", hgchefGeoHandle);
        ddds_[1] = &(hgchefGeoHandle->topology().dddConstants());
    }
    else
    {
        ddds_[1] = nullptr;
    }
    ddds_[2] = nullptr;
}

bool HGCalRecHitWorkerSimple::run(const edm::Event & evt, const HGCUncalibratedRecHit& uncalibRH,
        HGCRecHitCollection & result)
{
    DetId detid = uncalibRH.id();
// don't produce rechit if detid is a ghost one

    if((detid & rangeMask_) == rangeMatch_)
        return false;

    int thickness = -1;
    float sigmaNoiseGeV = 0.f;
    unsigned int layer = tools_->getLayerWithOffset(detid);
    HGCalDetId hid;
    float cce_correction = 1.0;

    switch (detid.subdetId())
    {
    case HGCEE:
        rechitMaker_->setADCToGeVConstant(float(hgceeUncalib2GeV_));
        hid = detid;
        thickness = ddds_[hid.subdetId() - 3]->waferTypeL(hid.wafer());
        cce_correction = hgcEE_cce_[thickness - 1];
        sigmaNoiseGeV = 1e-3 * weights_[layer] * rcorr_[thickness]
                    * hgcEE_noise_fC_[thickness - 1] / hgcEE_fCPerMIP_[thickness - 1];
        break;
    case HGCHEF:
        rechitMaker_->setADCToGeVConstant(float(hgchefUncalib2GeV_));
        hid = detid;
        thickness = ddds_[hid.subdetId() - 3]->waferTypeL(hid.wafer());
        cce_correction = hgcHEF_cce_[thickness - 1];
        sigmaNoiseGeV = 1e-3 * weights_[layer] * rcorr_[thickness]
                    * hgcHEF_noise_fC_[thickness - 1] / hgcHEF_fCPerMIP_[thickness - 1];
        break;
    case HcalEndcap:
    case HGCHEB:
        rechitMaker_->setADCToGeVConstant(float(hgchebUncalib2GeV_));
        hid = detid;
        sigmaNoiseGeV = 1e-3 * hgcHEB_noise_MIP_ * weights_[layer];
        break;
    default:
        throw cms::Exception("NonHGCRecHit") << "Rechit with detid = " << detid.rawId()
                << " is not HGC!";
    }

    // make the rechit and put in the output collection

    HGCRecHit myrechit(rechitMaker_->makeRecHit(uncalibRH, 0));
    const double new_E = myrechit.energy() * (thickness == -1 ? 1.0 : rcorr_[thickness])/cce_correction;


    myrechit.setEnergy(new_E); 
    myrechit.setSignalOverSigmaNoise(new_E/sigmaNoiseGeV);
    result.push_back(myrechit);

    return true;
}

HGCalRecHitWorkerSimple::~HGCalRecHitWorkerSimple()
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( HGCalRecHitWorkerFactory, HGCalRecHitWorkerSimple, "HGCalRecHitWorkerSimple" );
