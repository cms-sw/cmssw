#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "CondFormats/HcalObjects/interface/HcalRecoParams.h"
#include "CondFormats/DataRecord/interface/HcalRecoParamsRcd.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalChannelProperties.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalChannelPropertiesRecord.h"

class HcalChannelPropertiesEP : public edm::ESProducer {
public:
    typedef std::unique_ptr<HcalChannelPropertiesVec> ReturnType;

    inline HcalChannelPropertiesEP(const edm::ParameterSet&) 
    {
        auto cc = setWhatProduced(this);
        edm::ESInputTag qTag("", "withTopo");
        cc.setConsumes(condToken_).setConsumes(topoToken_).setConsumes(paramsToken_);
        cc.setConsumes(sevToken_).setConsumes(qualToken_, qTag).setConsumes(geomToken_);
    }

    inline virtual ~HcalChannelPropertiesEP() override {}

    ReturnType produce(const HcalChannelPropertiesRecord& rcd)
    {
        // There appears to be no easy way to trace the internal
        // dependencies of HcalDbService. So, rebuild the product
        // every time anything changes in the parent records.
        // This means that we are sometimes going to rebuild the
        // whole table on the lumi block boundaries instead of
        // just updating the list of bad channels.
        using namespace edm;

        // Retrieve various event setup records and data products
        const HcalDbRecord& dbRecord = rcd.getRecord<HcalDbRecord>();
        const HcalDbService& cond = dbRecord.get(condToken_);
        const HcalTopology& htopo = dbRecord.getRecord<HcalRecNumberingRecord>().get(topoToken_);
        const HcalRecoParams& params = dbRecord.getRecord<HcalRecoParamsRcd>().get(paramsToken_);
        const HcalSeverityLevelComputer& severity = rcd.getRecord<HcalSeverityLevelComputerRcd>().get(sevToken_);
        const HcalChannelQuality& qual = dbRecord.getRecord<HcalChannelQualityRcd>().get(qualToken_);
        const CaloGeometry& geom = rcd.getRecord<CaloGeometryRecord>().get(geomToken_);

        paramTS_ = std::make_unique<HcalRecoParams>(params);
        paramTS_->setTopo(&htopo);

        // Build the product
        ReturnType prod(new HcalChannelPropertiesVec(htopo.ncells()));
        std::array<HcalPipelinePedestalAndGain, 4> pedsAndGains;
        const HcalSubdetector subdetectors[3] = {HcalBarrel, HcalEndcap, HcalForward};

        for (HcalSubdetector subd : subdetectors) {
            const HcalGeometry* hcalGeom = static_cast<const HcalGeometry*>(
                geom.getSubdetectorGeometry(DetId::Hcal, subd));
            const std::vector<DetId>& ids = hcalGeom->getValidDetIds(DetId::Hcal, subd);

            for (const auto cell : ids) {
                const auto rawId = cell.rawId();

                // ADC decoding tools, etc
                const HcalRecoParam* param_ts = paramTS_->getValues(rawId);
                const HcalQIECoder* channelCoder = cond.getHcalCoder(cell);
                const HcalQIEShape* shape = cond.getHcalShape(channelCoder);
                const HcalSiPMParameter* siPMParameter = cond.getHcalSiPMParameter(cell);

                // Pedestals and gains
                const HcalCalibrations& calib = cond.getHcalCalibrations(cell);
                const HcalCalibrationWidths& calibWidth = cond.getHcalCalibrationWidths(cell);
                for (int capid=0; capid<4; ++capid) {
                    pedsAndGains[capid] = HcalPipelinePedestalAndGain(
                        calib.pedestal(capid), calibWidth.pedestal(capid),
                        calib.effpedestal(capid), calibWidth.effpedestal(capid),
                        calib.respcorrgain(capid), calibWidth.gain(capid));
                }

                // Channel quality
                const HcalChannelStatus* digistatus = qual.getValues(rawId);
                const bool taggedBadByDb = severity.dropChannel(digistatus->getValue());

                // Fill the table entry
                const unsigned linearId = htopo.detId2denseId(cell);
                prod->at(linearId) = HcalChannelProperties(
                    &calib, param_ts, channelCoder, shape,
                    siPMParameter, pedsAndGains, taggedBadByDb);
            }
        }

        return prod;
    }

private:
    HcalChannelPropertiesEP() = delete;
    HcalChannelPropertiesEP(const HcalChannelPropertiesEP&) = delete;
    HcalChannelPropertiesEP& operator=(const HcalChannelPropertiesEP&) = delete;

    edm::ESGetToken<HcalDbService, HcalDbRecord> condToken_;
    edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> topoToken_;
    edm::ESGetToken<HcalRecoParams, HcalRecoParamsRcd> paramsToken_;
    edm::ESGetToken<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd> sevToken_;
    edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> qualToken_;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;

    std::unique_ptr<HcalRecoParams> paramTS_;
};

DEFINE_FWK_EVENTSETUP_MODULE(HcalChannelPropertiesEP);
