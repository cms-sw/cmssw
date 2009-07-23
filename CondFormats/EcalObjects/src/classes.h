
#include <boost/cstdint.hpp>

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"
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
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
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


namespace{
  struct dictionary {
    uint32_t i32;
 
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
    std::map<uint32_t, float> dcuTempMap;
 
    EcalPTMTemperatures ptmTemperatures;
    std::map<uint32_t, float> ptmTempMap;
 
    EcalChannelStatus channelStatus;
 
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
 
    std::map< uint32_t, uint32_t> EcalTPGFineGrainTowerEEMap;
 
    std::map< uint32_t, uint16_t> EcalTPGTowerStatusMap;
 
    std::map<uint32_t, uint32_t> EcalTPGGroupsMap;
 
    EcalTPGLinearizationConst tpglinconstmap;
 
    EcalTPGLut lut;
    std::map< uint32_t, EcalTPGLut > EcalTPGLutMap;
    std::pair< const uint32_t, EcalTPGLut > EcalTPGLutMap_valuetype;
 
    EcalTPGPedestals tpgpedmap;
 
    EcalTPGWeights weights;
    std::map<uint32_t, EcalTPGWeights> EcalTPGWeightMap;
    std::pair<const uint32_t, EcalTPGWeights> EcalTPGWeightMap_valuetype;
 
    std::map<uint32_t, uint32_t> EcalTPGSlidingWindowMap;
 
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
 
    EcalTPGCrystalStatus tpgCrystalStatus;
  };
}
