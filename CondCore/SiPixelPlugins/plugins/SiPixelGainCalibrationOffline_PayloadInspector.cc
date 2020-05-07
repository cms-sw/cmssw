/*!
  \file SiPixelGainCalibrationOffline_PayloadInspector
  \Payload Inspector Plugin for SiPixel Gain Calibration Offline
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2020/04/01 11:31:00 $
*/

#include "CondCore/SiPixelPlugins/interface/SiPixelGainCalibHelper.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"

namespace {

  using SiPixelGainCalibrationOfflineGainsValues =
      gainCalibHelper::SiPixelGainCalibrationValues<gainCalibHelper::gainCalibPI::t_gain, SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibrationOfflinePedestalsValues =
      gainCalibHelper::SiPixelGainCalibrationValues<gainCalibHelper::gainCalibPI::t_pedestal,
                                                    SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibrationOfflineCorrelations =
      gainCalibHelper::SiPixelGainCalibrationCorrelations<SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibrationOfflineGainsByPart =
      gainCalibHelper::SiPixelGainCalibrationValuesByPart<gainCalibHelper::gainCalibPI::t_gain,
                                                          SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibrationOfflinePedestalsByPart =
      gainCalibHelper::SiPixelGainCalibrationValuesByPart<gainCalibHelper::gainCalibPI::t_pedestal,
                                                          SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainComparisonSingleTag =
      gainCalibHelper::SiPixelGainCalibrationValueComparisonSingleTag<gainCalibHelper::gainCalibPI::t_gain,
                                                                      SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibOfflinePedestalComparisonSingleTag =
      gainCalibHelper::SiPixelGainCalibrationValueComparisonSingleTag<gainCalibHelper::gainCalibPI::t_pedestal,
                                                                      SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainComparisonTwoTags =
      gainCalibHelper::SiPixelGainCalibrationValueComparisonTwoTags<gainCalibHelper::gainCalibPI::t_gain,
                                                                    SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibOfflinePedestalComparisonTwoTags =
      gainCalibHelper::SiPixelGainCalibrationValueComparisonTwoTags<gainCalibHelper::gainCalibPI::t_pedestal,
                                                                    SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainsBPIXMap =
      gainCalibHelper::SiPixelGainCalibrationBPIXMap<gainCalibHelper::gainCalibPI::t_gain,
                                                     SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibOfflinePedestalsBPIXMap =
      gainCalibHelper::SiPixelGainCalibrationBPIXMap<gainCalibHelper::gainCalibPI::t_pedestal,
                                                     SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainsFPIXMap =
      gainCalibHelper::SiPixelGainCalibrationFPIXMap<gainCalibHelper::gainCalibPI::t_gain,
                                                     SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibOfflinePedestalsFPIXMap =
      gainCalibHelper::SiPixelGainCalibrationFPIXMap<gainCalibHelper::gainCalibPI::t_pedestal,
                                                     SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainByRegionComparisonSingleTag =
      gainCalibHelper::SiPixelGainCalibrationByRegionComparisonSingleTag<gainCalibHelper::gainCalibPI::t_gain,
                                                                         SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibOfflinePedestalByRegionComparisonSingleTag =
      gainCalibHelper::SiPixelGainCalibrationByRegionComparisonSingleTag<gainCalibHelper::gainCalibPI::t_pedestal,
                                                                         SiPixelGainCalibrationOffline>;

  using SiPixelGainCalibOfflineGainByRegionComparisonTwoTags =
      gainCalibHelper::SiPixelGainCalibrationByRegionComparisonTwoTags<gainCalibHelper::gainCalibPI::t_gain,
                                                                       SiPixelGainCalibrationOffline>;
  using SiPixelGainCalibOfflinePedestalByRegionComparisonTwoTags =
      gainCalibHelper::SiPixelGainCalibrationByRegionComparisonTwoTags<gainCalibHelper::gainCalibPI::t_pedestal,
                                                                       SiPixelGainCalibrationOffline>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiPixelGainCalibrationOffline) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflineGainsValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflinePedestalsValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflineCorrelations);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflineGainsByPart);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationOfflinePedestalsByPart);
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
}
