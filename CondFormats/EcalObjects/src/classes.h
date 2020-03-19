
#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalCondTowerObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalPulseSymmCovariances.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalDCUTemperatures.h"
#include "CondFormats/EcalObjects/interface/EcalPTMTemperatures.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
#include "CondFormats/EcalObjects/interface/EcalDAQStatusCode.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "CondFormats/EcalObjects/interface/EcalTimeDependentCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "CondFormats/EcalObjects/interface/EcalLinearCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainConstEB.h"
#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"
#include "CondFormats/EcalObjects/interface/EcalFunctionParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatusCode.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"
#include "CondFormats/EcalObjects/interface/EcalSimPulseShape.h"
#include <cstdint>

namespace CondFormats_EcalObjects {
  struct dictionary {
    std::vector<EcalChannelStatusCode> v_ecalChannelStatusCode;
    EcalContainer<EEDetId, EcalChannelStatusCode> ec_eeDetId_ecalChannelStatusCode;
    EcalContainer<EBDetId, EcalChannelStatusCode> ec_ebDetId_ecalChannelStatusCode;
    EcalCondObjectContainer<EcalChannelStatusCode> channelStatus;  //typedef: EcalChannelStatus

    std::vector<EcalDQMStatusCode> v_ecalDQMStatusCode;
    EcalContainer<EEDetId, EcalDQMStatusCode> ec_eeDetId_ecalDQMStatusCode;
    EcalContainer<EBDetId, EcalDQMStatusCode> ec_ebDetId_ecalDQMStatusCode;
    EcalCondObjectContainer<EcalDQMStatusCode> dqmChannelStatus;  //typedef EcalDQMChannelStatus

    std::vector<EcalMGPAGainRatio> v_ecalMGPAGainRatio;
    EcalContainer<EEDetId, EcalMGPAGainRatio> ec_eeDetId_ecalMGPAGainRatio;
    EcalContainer<EBDetId, EcalMGPAGainRatio> ec_ebDetId_ecalMGPAGainRatio;
    EcalCondObjectContainer<EcalMGPAGainRatio> gainratios;  // typedef EcalGainRatios

    std::vector<EcalMappingElement> v_ecalMappingElement;
    EcalContainer<EEDetId, EcalMappingElement> ec_eeDetId_ecalMappingElement;
    EcalContainer<EBDetId, EcalMappingElement> ec_ebDetId_ecalMappingElement;
    EcalCondObjectContainer<EcalMappingElement> ecalMap;  //typedef EcalMappingElectronics

    std::vector<EcalPedestal> v_ecalPedestal;
    EcalContainer<EEDetId, EcalPedestal> ec_eeDetId_ecalPedestal;
    EcalContainer<EBDetId, EcalPedestal> ec_ebDetId_ecalPedestal;
    EcalCondObjectContainer<EcalPedestal> pedmap;  //typedef EcalPedestals

    std::vector<EcalTPGCrystalStatusCode> v_ecalTPGCrystalStatusCode;
    EcalContainer<EEDetId, EcalTPGCrystalStatusCode> ec_eeDetId_ecalTPGCrystalStatusCode;
    EcalContainer<EBDetId, EcalTPGCrystalStatusCode> ec_ebDetId_ecalTPGCrystalStatusCode;
    EcalCondObjectContainer<EcalTPGCrystalStatusCode> tpgCrystalStatus;  //typedef EcalTPGCrystalStatus

    std::vector<EcalTPGLinearizationConstant> v_ecalTPGLinearizationConstant;
    EcalContainer<EEDetId, EcalTPGLinearizationConstant> ec_eeDetId_ecalTPGLinearizationConstant;
    EcalContainer<EBDetId, EcalTPGLinearizationConstant> ec_ebDetId_ecalTPGLinearizationConstant;
    EcalCondObjectContainer<EcalTPGLinearizationConstant> tpglinconstmap;  //typedef EcalTPGLinearizationConst

    std::vector<EcalTPGPedestal> v_ecalTPGPedestal;
    EcalContainer<EEDetId, EcalTPGPedestal> ec_eeDetId_ecalTPGPedestal;
    EcalContainer<EBDetId, EcalTPGPedestal> ec_ebDetId_ecalTPGPedestal;
    EcalCondObjectContainer<EcalTPGPedestal> tpgpedmap;  //typedef EcalTPGPedestals

    std::vector<EcalXtalGroupId> v_ecalXtalGroupId;
    EcalContainer<EEDetId, EcalXtalGroupId> ec_eeDetId_ecalXtalGroupId;
    EcalContainer<EBDetId, EcalXtalGroupId> ec_ebDetId_ecalXtalGroupId;
    EcalCondObjectContainer<EcalXtalGroupId> gg;  //typedef EcalWeightXtalGroups

    EcalContainer<EEDetId, float> ec_eeDetId_float;
    EcalContainer<EBDetId, float> ec_ebDetId_float;
    EcalCondObjectContainer<float>
        floatCondObjectContainer;  //typedefs: EcalFloatCondObjectContainer, EcalLinearCorrections, EcalIntercalibConstants, EcalIntercalibConstantsMC, EcalIntercalibErrors, EcalLaserAPDPNRatiosRef, EcalLaserAlphas, EcalTimeCalibConstants, EcalTimeCalibErrors, EcalPFRecHitThresholds

    EcalLaserAPDPNRatios laser_map;
    std::vector<EcalLaserAPDPNRatios::EcalLaserAPDPNpair> laser_pair_map;
    std::vector<EcalLaserAPDPNRatios::EcalLaserTimeStamp> laser_time_map;
    EcalContainer<EEDetId, EcalLaserAPDPNRatios::EcalLaserAPDPNpair> laser_ec_eeDetId_pair;
    EcalContainer<EBDetId, EcalLaserAPDPNRatios::EcalLaserAPDPNpair> laser_ec_ebDetId_pair;
    EcalCondObjectContainer<EcalLaserAPDPNRatios::EcalLaserAPDPNpair> laser_map_dm;

    EcalTimeDependentCorrections correction_map;
    std::vector<EcalTimeDependentCorrections::Values> value_map;
    std::vector<EcalTimeDependentCorrections::Times> time_map;
    EcalContainer<EEDetId, EcalTimeDependentCorrections::Values> ec_eeDetId_pair;
    EcalContainer<EBDetId, EcalTimeDependentCorrections::Values> ec_ebDetId_pair;
    EcalCondObjectContainer<EcalTimeDependentCorrections::Values> correction_map_dm;

    EcalLinearCorrections linear_correction_map;

    EcalContainer<EcalTrigTowerDetId, EcalChannelStatusCode> ec_ettDetId_ecalChannelStatusCode;
    EcalContainer<EcalScDetId, EcalChannelStatusCode> ec_esDetId_ecalChannelStatusCode;
    EcalCondTowerObjectContainer<EcalChannelStatusCode> dcsTowerStatus;  //typedef EcalDCSTowerStatus

    EcalContainer<EcalTrigTowerDetId, EcalDAQStatusCode> ec_ettDetId_ecalDAQStatusCode;
    EcalContainer<EcalScDetId, EcalDAQStatusCode> ec_esDetId_ecalDAQStatusCode;
    EcalCondTowerObjectContainer<EcalDAQStatusCode> daqTowerStatus;  //typedef EcalDAQTowerStatus

    EcalContainer<EcalTrigTowerDetId, EcalDQMStatusCode> ec_ettDetId_ecalDQMStatusCode;
    EcalContainer<EcalScDetId, EcalDQMStatusCode> ec_esDetId_ecalDQMStatusCode;
    EcalCondTowerObjectContainer<EcalDQMStatusCode> dqmTowerStatus;  //typedef EcalDQMTowerStatus

    EcalTBWeights tbwgt;
    EcalWeightSet wset;
    EcalTBWeights::EcalTDCId id;
    std::pair<EcalXtalGroupId, EcalTBWeights::EcalTDCId> wgpair;
    std::map<std::pair<EcalXtalGroupId, EcalTBWeights::EcalTDCId>, EcalWeightSet> wgmap;
    std::pair<const std::pair<EcalXtalGroupId, EcalTBWeights::EcalTDCId>, EcalWeightSet> wgmapvalue;

    EcalSampleMask sampleMask;

    EcalADCToGeVConstant adcfactor;

    EcalTimeOffsetConstant timeOffsetConstant;

    EcalDCUTemperatures dcuTemperatures;

    EcalPTMTemperatures ptmTemperatures;

    EcalTPGFineGrainConstEB grain;
    std::map<uint32_t, EcalTPGFineGrainConstEB> EcalTPGFineGrainEBMap;
    std::pair<const uint32_t, EcalTPGFineGrainConstEB> EcalTPGFineGrainEBMap_valuetype;

    std::map<uint32_t, EcalTPGFineGrainStripEE::Item> EcalTPGFineGrainStripEEMap;
    std::pair<const uint32_t, EcalTPGFineGrainStripEE::Item> EcalTPGFineGrainStripEEMap_valuetype;

    EcalTPGLut lut;
    std::map<uint32_t, EcalTPGLut> EcalTPGLutMap;
    std::pair<const uint32_t, EcalTPGLut> EcalTPGLutMap_valuetype;

    EcalTPGWeights weightsweights;
    std::map<uint32_t, EcalTPGWeights> EcalTPGWeightMap;
    std::pair<const uint32_t, EcalTPGWeights> EcalTPGWeightMap_valuetype;

    EcalFunParams
        funParams;  // typdefs: EcalClusterCrackCorrParameters, EcalClusterEnergyCorrectionObjectSpecificParameters, EcalClusterEnergyCorrectionParameters, EcalClusterEnergyUncertaintyParameters, EcalClusterLocalContCorrParameters

    EcalTPGFineGrainEBGroup fgrgroup;

    EcalTPGLutGroup lutgroup;

    EcalTPGWeightGroup wgroup;

    EcalTPGPhysicsConst::Item foo1;
    std::map<uint32_t, EcalTPGPhysicsConst::Item> phConst;
    std::pair<const uint32_t, EcalTPGPhysicsConst::Item> phConst_valuetype;

    EcalTPGTowerStatus towerstatus;
    std::map<uint32_t, uint16_t> tStatus;
    std::pair<const uint32_t, uint16_t> tStatus_valuetype;

    EcalTPGTowerStatus stripstatus;

    EcalTPGTowerStatus spike;

    EcalSRSettings ecalSRSettings;

    EcalTimeBiasCorrections timeBiasCorrections;

    EcalSamplesCorrelation samplesCorrelation;

    std::vector<EcalPulseShape> v_ecalPulseShape;
    EcalContainer<EEDetId, EcalPulseShape> ec_eeDetId_ecalPulseShape;
    EcalContainer<EBDetId, EcalPulseShape> ec_ebDetId_ecalPulseShape;
    EcalCondObjectContainer<EcalPulseShape> ecalPSmap;  //typedef EcalPulseShape

    std::vector<EcalPulseCovariance> v_ecalPulseCovariance;
    EcalContainer<EEDetId, EcalPulseCovariance> ec_eeDetId_ecalPulseCovariance;
    EcalContainer<EBDetId, EcalPulseCovariance> ec_ebDetId_ecalPulseCovariance;
    EcalCondObjectContainer<EcalPulseCovariance> ecalPCmap;  //typedef EcalPulseCovariance

    std::vector<EcalPulseSymmCovariance> v_ecalPulseSymmCovariance;
    EcalContainer<EEDetId, EcalPulseSymmCovariance> ec_eeDetId_ecalPulseSymmCovariance;
    EcalContainer<EBDetId, EcalPulseSymmCovariance> ec_ebDetId_ecalPulseSymmCovariance;
    EcalCondObjectContainer<EcalPulseSymmCovariance> ecalSPCmap;  //typedef EcalPulseSymmCovariance

    EcalSimPulseShape ecal_sim_pulse_shapes;
  };
}  // namespace CondFormats_EcalObjects
