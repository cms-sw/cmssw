/*!
  \file SiPixelGainCalibrationOffline_PayloadInspector
  \Payload Inspector Plugin for SiPixel Gain Calibration Offline
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2020/04/01 11:31:00 $
*/

#include "CondCore/SiPixelPlugins/interface/SiPixelGainCalibHelper.h"

namespace {

  using namespace gainCalibHelper;

  using SiPixelGainCalibrationOfflineGainsValues =
      SiPixelGainCalibrationValues<gainCalibPI::t_gain, SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibrationOfflinePedestalsValues =
      SiPixelGainCalibrationValues<gainCalibPI::t_pedestal, SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibrationOfflineGainsValuesBarrel =
      SiPixelGainCalibrationValuesPerRegion<true, gainCalibPI::t_gain, SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibrationOfflineGainsValuesEndcap =
      SiPixelGainCalibrationValuesPerRegion<false, gainCalibPI::t_gain, SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibrationOfflinePedestalsValuesBarrel =
      SiPixelGainCalibrationValuesPerRegion<true, gainCalibPI::t_pedestal, SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibrationOfflinePedestalsValuesEndcap =
      SiPixelGainCalibrationValuesPerRegion<false, gainCalibPI::t_pedestal, SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibrationOfflineCorrelations = SiPixelGainCalibrationCorrelations<SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibrationOfflineGainsByPart =
      SiPixelGainCalibrationValuesByPart<gainCalibPI::t_gain, SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibrationOfflinePedestalsByPart =
      SiPixelGainCalibrationValuesByPart<gainCalibPI::t_pedestal, SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainComparisonSingleTag =
      SiPixelGainCalibrationValueComparisonSingleTag<gainCalibPI::t_gain, SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibOfflinePedestalComparisonSingleTag =
      SiPixelGainCalibrationValueComparisonSingleTag<gainCalibPI::t_pedestal, SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainComparisonTwoTags =
      SiPixelGainCalibrationValueComparisonTwoTags<gainCalibPI::t_gain, SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibOfflinePedestalComparisonTwoTags =
      SiPixelGainCalibrationValueComparisonTwoTags<gainCalibPI::t_pedestal, SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainComparisonBarrelSingleTag =
      SiPixelGainCalibrationValuesComparisonPerRegion<true,
                                                      gainCalibPI::t_gain,
                                                      cond::payloadInspector::MULTI_IOV,
                                                      1,
                                                      SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflinePedestalComparisonBarrelSingleTag =
      SiPixelGainCalibrationValuesComparisonPerRegion<true,
                                                      gainCalibPI::t_pedestal,
                                                      cond::payloadInspector::MULTI_IOV,
                                                      1,
                                                      SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainComparisonBarrelTwoTags =
      SiPixelGainCalibrationValuesComparisonPerRegion<true,
                                                      gainCalibPI::t_gain,
                                                      cond::payloadInspector::SINGLE_IOV,
                                                      2,
                                                      SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflinePedestalComparisonBarrelTwoTags =
      SiPixelGainCalibrationValuesComparisonPerRegion<true,
                                                      gainCalibPI::t_pedestal,
                                                      cond::payloadInspector::SINGLE_IOV,
                                                      2,
                                                      SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainComparisonEndcapSingleTag =
      SiPixelGainCalibrationValuesComparisonPerRegion<false,
                                                      gainCalibPI::t_gain,
                                                      cond::payloadInspector::MULTI_IOV,
                                                      1,
                                                      SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflinePedestalComparisonEndcapSingleTag =
      SiPixelGainCalibrationValuesComparisonPerRegion<false,
                                                      gainCalibPI::t_pedestal,
                                                      cond::payloadInspector::MULTI_IOV,
                                                      1,
                                                      SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainComparisonEndcapTwoTags =
      SiPixelGainCalibrationValuesComparisonPerRegion<false,
                                                      gainCalibPI::t_gain,
                                                      cond::payloadInspector::SINGLE_IOV,
                                                      2,
                                                      SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflinePedestalComparisonEndcapTwoTags =
      SiPixelGainCalibrationValuesComparisonPerRegion<false,
                                                      gainCalibPI::t_pedestal,
                                                      cond::payloadInspector::SINGLE_IOV,
                                                      2,
                                                      SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainsBPIXMap =
      SiPixelGainCalibrationMap<gainCalibPI::t_gain, SiPixelGainCalibrationOffline, SiPixelPI::t_barrel>;

  using SiPixelGainCalibOfflinePedestalsBPIXMap =
      SiPixelGainCalibrationMap<gainCalibPI::t_pedestal, SiPixelGainCalibrationOffline, SiPixelPI::t_barrel>;

  using SiPixelGainCalibOfflineGainsFPIXMap =
      SiPixelGainCalibrationMap<gainCalibPI::t_gain, SiPixelGainCalibrationOffline, SiPixelPI::t_forward>;

  using SiPixelGainCalibOfflinePedestalsFPIXMap =
      SiPixelGainCalibrationMap<gainCalibPI::t_pedestal, SiPixelGainCalibrationOffline, SiPixelPI::t_forward>;

  using SiPixelGainCalibOfflineGainByRegionComparisonSingleTag =
      SiPixelGainCalibrationByRegionComparisonBase<gainCalibPI::t_gain,
                                                   SiPixelGainCalibrationOffline,
                                                   cond::payloadInspector::MULTI_IOV,
                                                   1>;
  using SiPixelGainCalibOfflinePedestalByRegionComparisonSingleTag =
      SiPixelGainCalibrationByRegionComparisonBase<gainCalibPI::t_pedestal,
                                                   SiPixelGainCalibrationOffline,
                                                   cond::payloadInspector::MULTI_IOV,
                                                   1>;

  using SiPixelGainCalibOfflineGainByRegionComparisonTwoTags =
      SiPixelGainCalibrationByRegionComparisonBase<gainCalibPI::t_gain,
                                                   SiPixelGainCalibrationOffline,
                                                   cond::payloadInspector::SINGLE_IOV,
                                                   2>;
  using SiPixelGainCalibOfflinePedestalByRegionComparisonTwoTags =
      SiPixelGainCalibrationByRegionComparisonBase<gainCalibPI::t_pedestal,
                                                   SiPixelGainCalibrationOffline,
                                                   cond::payloadInspector::SINGLE_IOV,
                                                   2>;

  using SiPixelGainCalibOfflineGainDiffRatioTwoTags =
      SiPixelGainCalibDiffAndRatioBase<gainCalibPI::t_gain,
                                       cond::payloadInspector::SINGLE_IOV,
                                       2,
                                       SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflinePedestalDiffRatioTwoTags =
      SiPixelGainCalibDiffAndRatioBase<gainCalibPI::t_pedestal,
                                       cond::payloadInspector::SINGLE_IOV,
                                       2,
                                       SiPixelGainCalibrationOffline>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiPixelGainCalibrationOffline) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflineGainsValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflinePedestalsValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflineGainsValuesBarrel);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflineGainsValuesEndcap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflinePedestalsValuesBarrel);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflinePedestalsValuesEndcap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflineCorrelations);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflineGainsByPart);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflinePedestalsByPart);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainComparisonBarrelSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainComparisonBarrelTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalComparisonBarrelSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalComparisonBarrelTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainComparisonEndcapSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainComparisonEndcapTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalComparisonEndcapSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalComparisonEndcapTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainComparisonSingleTag)
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainsBPIXMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalsBPIXMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainsFPIXMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalsFPIXMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainByRegionComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalByRegionComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainByRegionComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalByRegionComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflineGainDiffRatioTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibOfflinePedestalDiffRatioTwoTags);
}
