#include "CondFormats/Common/interface/SerializationTest.h"

#include "CondFormats/SiPixelObjects/interface/Serialization.h"

int main()
{
    testSerialization<PixelDCSObject<bool>>();
    testSerialization<PixelDCSObject<float>>();
    testSerialization<PixelDCSObject<CaenChannel>>();
    testSerialization<SiPixelCPEGenericErrorParm>();
    testSerialization<SiPixelCPEGenericErrorParm::DbEntry>();
    testSerialization<SiPixelCPEGenericErrorParm::DbEntryBinSize>();
    testSerialization<SiPixelCalibConfiguration>();
    testSerialization<SiPixelDbItem>();
    //testSerialization<SiPixelFedCabling>(); abstract
    testSerialization<SiPixelFedCablingMap>();
    testSerialization<SiPixelFedCablingMap::Key>();
    testSerialization<SiPixelGainCalibration>();
    testSerialization<SiPixelGainCalibration::DetRegistry>();
    testSerialization<SiPixelGainCalibrationForHLT>();
    testSerialization<SiPixelGainCalibrationForHLT::DetRegistry>();
    testSerialization<SiPixelGainCalibrationOffline>();
    testSerialization<SiPixelGainCalibrationOffline::DetRegistry>();
    testSerialization<SiPixelLorentzAngle>();
    testSerialization<SiPixelPedestals>();
    testSerialization<SiPixelPerformanceSummary>();
    testSerialization<SiPixelPerformanceSummary::DetSummary>();
    testSerialization<SiPixelQuality>();
    testSerialization<SiPixelQuality::disabledModuleType>();
    testSerialization<SiPixelTemplateDBObject>();
    testSerialization<sipixelobjects::PixelROC>();
    testSerialization<std::map<SiPixelFedCablingMap::Key,sipixelobjects::PixelROC>>();
    testSerialization<std::map<int,std::vector<SiPixelDbItem>>>();
    testSerialization<std::pair<const SiPixelFedCablingMap::Key,sipixelobjects::PixelROC>>();
    testSerialization<std::vector<SiPixelCPEGenericErrorParm::DbEntry>>();
    testSerialization<std::vector<SiPixelCPEGenericErrorParm::DbEntryBinSize>>();
    testSerialization<std::vector<SiPixelDbItem>>();
    testSerialization<std::vector<SiPixelGainCalibration::DetRegistry>>();
    testSerialization<std::vector<SiPixelGainCalibrationForHLT::DetRegistry>>();
    testSerialization<std::vector<SiPixelGainCalibrationOffline::DetRegistry>>();
    testSerialization<std::vector<SiPixelPerformanceSummary::DetSummary>>();
    testSerialization<std::vector<SiPixelQuality::disabledModuleType>>();

    return 0;
}
