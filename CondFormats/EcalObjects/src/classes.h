
#include <boost/cstdint.hpp>

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalCondTowerObjectContainer.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"
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
#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
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
#include "CondFormats/EcalObjects/interface/EcalClusterLocalContCorrParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterCrackCorrParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyUncertaintyParameters.h"
#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGLutGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatusCode.h"

#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"

#include "CondFormats/EcalObjects/interface/EcalAlignment.h"

namespace{
  struct dictionary {
    EcalPedestals pedmap;
    EcalWeightXtalGroups gg;
 
    EcalTBWeights tbwgt;
    EcalWeightSet wset;
    EcalTBWeights::EcalTDCId id;
    std::pair< EcalXtalGroupId, EcalTBWeights::EcalTDCId > wgpair;
    std::map< std::pair< EcalXtalGroupId, EcalTBWeights::EcalTDCId > , EcalWeightSet > wgmap;
    std::pair< const std::pair< EcalXtalGroupId, EcalTBWeights::EcalTDCId > , EcalWeightSet > wgmapvalue;
 
    EcalADCToGeVConstant adcfactor;
 
    EcalGainRatios gainratios;
 
    EcalIntercalibConstants intercalib;
    EcalIntercalibConstantsMC intercalibMC;
    EcalIntercalibErrors intercalibErrors;
 
    EcalTimeCalibConstants timeCalib;
    EcalTimeCalibErrors timeCalibErrors;
 
    EcalDCUTemperatures dcuTemperatures;
 
    EcalPTMTemperatures ptmTemperatures;
 
    EcalChannelStatus channelStatus;
    EcalDQMChannelStatus dqmChannelStatus;

    EcalDQMTowerStatus dqmTowerStatus;
    EcalDCSTowerStatus dcsTowerStatus;
    EcalDAQTowerStatus daqTowerStatus;
 
    EcalLaserAlphas laserAplhas;
 
    EcalLaserAPDPNRatios laser_map;
    EcalCondObjectContainer<EcalLaserAPDPNRatios::EcalLaserAPDPNpair> laser_map_dm;
    std::vector<EcalLaserAPDPNRatios::EcalLaserTimeStamp> time_map ;
 
    EcalLaserAPDPNRatiosRef laserAPDPNRatiosRef;
 
    EcalTPGFineGrainConstEB grain;
    std::map<uint32_t, EcalTPGFineGrainConstEB> EcalTPGFineGrainEBMap ;
    std::pair<const uint32_t, EcalTPGFineGrainConstEB> EcalTPGFineGrainEBMap_valuetype ;
 
    std::map< uint32_t, EcalTPGFineGrainStripEE::Item > EcalTPGFineGrainStripEEMap;
    std::pair< const uint32_t, EcalTPGFineGrainStripEE::Item > EcalTPGFineGrainStripEEMap_valuetype;
 
 
    EcalTPGLinearizationConst tpglinconstmap;
 
    EcalTPGLut lut;
    std::map< uint32_t, EcalTPGLut > EcalTPGLutMap;
    std::pair< const uint32_t, EcalTPGLut > EcalTPGLutMap_valuetype;
 
    EcalTPGPedestals tpgpedmap;
 
    EcalTPGWeights weightsweights;
    std::map<uint32_t, EcalTPGWeights> EcalTPGWeightMap;
    std::pair<const uint32_t, EcalTPGWeights> EcalTPGWeightMap_valuetype;
 
    EcalMappingElectronics ecalMap;
 
    EcalClusterLocalContCorrParameters clusterLocalContCorrParams;
 
    EcalClusterCrackCorrParameters clusterCrackCorrParams;
 
    EcalClusterEnergyCorrectionParameters clusterEnergyCorrectionParams;

    EcalClusterEnergyUncertaintyParameters clusterEnergyUncertaintyParams;
 
    EcalTPGFineGrainEBGroup fgrgroup;
 
    EcalTPGLutGroup lutgroup;
 
    EcalTPGWeightGroup wgroup;
 
    EcalTPGPhysicsConst::Item foo1;
    std::map< uint32_t, EcalTPGPhysicsConst::Item >  phConst;
    std::pair< const uint32_t, EcalTPGPhysicsConst::Item >  phConst_valuetype;
 
    EcalTPGTowerStatus towerstatus;
    std::map< uint32_t, uint16_t > tStatus;
    std::pair< const uint32_t, uint16_t > tStatus_valuetype;

    
    EcalTPGCrystalStatus tpgCrystalStatus;

    EcalSRSettings ecalSRSettings;
    std::vector<std::vector<short> > ecalSRSettings_srpMasksFromConfig;
    std::vector<std::vector<float> > ecalSRSettings_dccNormalizedWeights_0;
    //    std::vector<float> ecalSRSettings_dccNormalizedWeights_1;
    //    float ecalSRSettings_dccNormalizedWeights_elt_2;

    EcalAlignment alignment;
    
  };
}
