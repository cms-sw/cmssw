/*!
  \file SiPixelGainCalibrationOffline_PayloadInspector
  \Payload Inspector Plugin for SiPixel Gain Calibration for HLT
  \author M. Musich
  \version $Revision: 1.0 $
  \date $Date: 2020/04/01 11:31:00 $
*/

#include "CondCore/SiPixelPlugins/interface/SiPixelGainCalibHelper.h"

namespace {

  using namespace gainCalibHelper;

  using SiPixelGainCalibrationForHLTGainsValues =
      SiPixelGainCalibrationValues<gainCalibPI::t_gain, SiPixelGainCalibrationForHLT>;
  using SiPixelGainCalibrationForHLTPedestalsValues =
      SiPixelGainCalibrationValues<gainCalibPI::t_pedestal, SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibrationForHLTGainsValuesBarrel =
      SiPixelGainCalibrationValuesPerRegion<true, gainCalibPI::t_gain, SiPixelGainCalibrationForHLT>;
  using SiPixelGainCalibrationForHLTGainsValuesEndcap =
      SiPixelGainCalibrationValuesPerRegion<false, gainCalibPI::t_gain, SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibrationForHLTPedestalsValuesBarrel =
      SiPixelGainCalibrationValuesPerRegion<true, gainCalibPI::t_pedestal, SiPixelGainCalibrationForHLT>;
  using SiPixelGainCalibrationForHLTPedestalsValuesEndcap =
      SiPixelGainCalibrationValuesPerRegion<false, gainCalibPI::t_pedestal, SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibrationForHLTCorrelations = SiPixelGainCalibrationCorrelations<SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibrationForHLTGainsByPart =
      SiPixelGainCalibrationValuesByPart<gainCalibPI::t_gain, SiPixelGainCalibrationForHLT>;
  using SiPixelGainCalibrationForHLTPedestalsByPart =
      SiPixelGainCalibrationValuesByPart<gainCalibPI::t_pedestal, SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTGainComparisonSingleTag =
      SiPixelGainCalibrationValueComparisonSingleTag<gainCalibPI::t_gain, SiPixelGainCalibrationForHLT>;
  using SiPixelGainCalibForHLTPedestalComparisonSingleTag =
      SiPixelGainCalibrationValueComparisonSingleTag<gainCalibPI::t_pedestal, SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTGainComparisonTwoTags =
      SiPixelGainCalibrationValueComparisonTwoTags<gainCalibPI::t_gain, SiPixelGainCalibrationForHLT>;
  using SiPixelGainCalibForHLTPedestalComparisonTwoTags =
      SiPixelGainCalibrationValueComparisonTwoTags<gainCalibPI::t_pedestal, SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTGainComparisonBarrelSingleTag =
      SiPixelGainCalibrationValuesComparisonPerRegion<true,
                                                      gainCalibPI::t_gain,
                                                      cond::payloadInspector::MULTI_IOV,
                                                      1,
                                                      SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTPedestalComparisonBarrelSingleTag =
      SiPixelGainCalibrationValuesComparisonPerRegion<true,
                                                      gainCalibPI::t_pedestal,
                                                      cond::payloadInspector::MULTI_IOV,
                                                      1,
                                                      SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTGainComparisonBarrelTwoTags =
      SiPixelGainCalibrationValuesComparisonPerRegion<true,
                                                      gainCalibPI::t_gain,
                                                      cond::payloadInspector::SINGLE_IOV,
                                                      2,
                                                      SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTPedestalComparisonBarrelTwoTags =
      SiPixelGainCalibrationValuesComparisonPerRegion<true,
                                                      gainCalibPI::t_pedestal,
                                                      cond::payloadInspector::SINGLE_IOV,
                                                      2,
                                                      SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTGainComparisonEndcapSingleTag =
      SiPixelGainCalibrationValuesComparisonPerRegion<false,
                                                      gainCalibPI::t_gain,
                                                      cond::payloadInspector::MULTI_IOV,
                                                      1,
                                                      SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTPedestalComparisonEndcapSingleTag =
      SiPixelGainCalibrationValuesComparisonPerRegion<false,
                                                      gainCalibPI::t_pedestal,
                                                      cond::payloadInspector::MULTI_IOV,
                                                      1,
                                                      SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTGainComparisonEndcapTwoTags =
      SiPixelGainCalibrationValuesComparisonPerRegion<false,
                                                      gainCalibPI::t_gain,
                                                      cond::payloadInspector::SINGLE_IOV,
                                                      2,
                                                      SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTPedestalComparisonEndcapTwoTags =
      SiPixelGainCalibrationValuesComparisonPerRegion<false,
                                                      gainCalibPI::t_pedestal,
                                                      cond::payloadInspector::SINGLE_IOV,
                                                      2,
                                                      SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTGainsBPIXMap =
      SiPixelGainCalibrationMap<gainCalibPI::t_gain, SiPixelGainCalibrationForHLT, SiPixelPI::t_barrel>;
  using SiPixelGainCalibForHLTPedestalsBPIXMap =
      SiPixelGainCalibrationMap<gainCalibPI::t_pedestal, SiPixelGainCalibrationForHLT, SiPixelPI::t_barrel>;

  using SiPixelGainCalibForHLTGainsFPIXMap =
      SiPixelGainCalibrationMap<gainCalibPI::t_gain, SiPixelGainCalibrationForHLT, SiPixelPI::t_forward>;
  using SiPixelGainCalibForHLTPedestalsFPIXMap =
      SiPixelGainCalibrationMap<gainCalibPI::t_pedestal, SiPixelGainCalibrationForHLT, SiPixelPI::t_forward>;

  using SiPixelGainCalibForHLTGainByRegionComparisonSingleTag =
      SiPixelGainCalibrationByRegionComparisonBase<gainCalibPI::t_gain,
                                                   SiPixelGainCalibrationForHLT,
                                                   cond::payloadInspector::MULTI_IOV,
                                                   1>;
  using SiPixelGainCalibForHLTPedestalByRegionComparisonSingleTag =
      SiPixelGainCalibrationByRegionComparisonBase<gainCalibPI::t_pedestal,
                                                   SiPixelGainCalibrationForHLT,
                                                   cond::payloadInspector::MULTI_IOV,
                                                   1>;

  using SiPixelGainCalibForHLTGainByRegionComparisonTwoTags =
      SiPixelGainCalibrationByRegionComparisonBase<gainCalibPI::t_gain,
                                                   SiPixelGainCalibrationForHLT,
                                                   cond::payloadInspector::SINGLE_IOV,
                                                   2>;
  using SiPixelGainCalibForHLTPedestalByRegionComparisonTwoTags =
      SiPixelGainCalibrationByRegionComparisonBase<gainCalibPI::t_pedestal,
                                                   SiPixelGainCalibrationForHLT,
                                                   cond::payloadInspector::SINGLE_IOV,
                                                   2>;

  using SiPixelGainCalibForHLTGainDiffRatioTwoTags =
      SiPixelGainCalibDiffAndRatioBase<gainCalibPI::t_gain,
                                       cond::payloadInspector::SINGLE_IOV,
                                       2,
                                       SiPixelGainCalibrationForHLT>;

  using SiPixelGainCalibForHLTPedestalDiffRatioTwoTags =
      SiPixelGainCalibDiffAndRatioBase<gainCalibPI::t_pedestal,
                                       cond::payloadInspector::SINGLE_IOV,
                                       2,
                                       SiPixelGainCalibrationForHLT>;

}  // namespace

PAYLOAD_INSPECTOR_MODULE(SiPixelGainCalibrationForHLT) {
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationForHLTGainsValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationForHLTPedestalsValues);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationForHLTGainsValuesBarrel);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationForHLTGainsValuesEndcap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationForHLTPedestalsValuesBarrel);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationForHLTPedestalsValuesEndcap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationForHLTCorrelations);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationForHLTGainsByPart);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibrationForHLTPedestalsByPart);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainComparisonBarrelSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainComparisonBarrelTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalComparisonBarrelSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalComparisonBarrelTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainComparisonEndcapSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainComparisonEndcapTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalComparisonEndcapSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalComparisonEndcapTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainComparisonSingleTag)
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainsBPIXMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalsBPIXMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainsFPIXMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalsFPIXMap);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainByRegionComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalByRegionComparisonSingleTag);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainByRegionComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalByRegionComparisonTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTGainDiffRatioTwoTags);
  PAYLOAD_INSPECTOR_CLASS(SiPixelGainCalibForHLTPedestalDiffRatioTwoTags);
}
