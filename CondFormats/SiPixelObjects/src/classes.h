#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDbItem.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDisabledModules.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPedestals.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"
#include "CondFormats/SiPixelObjects/interface/SiPixel2DTemplateDBObject.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"

template class PixelDCSObject<bool>;
template class PixelDCSObject<float>;
template class PixelDCSObject<CaenChannel>;

namespace CondFormats_SiPixelObjects {
  struct dictionary {
    std::map<SiPixelFedCablingMap::Key, sipixelobjects::PixelROC> theMap; 
    std::pair<const SiPixelFedCablingMap::Key, sipixelobjects::PixelROC> theMapValueT; 
 
    std::map< unsigned int, SiPixelPedestals::SiPixelPedestalsVector> sipixped;
 
    std::vector<char>::iterator p1;
    std::vector<char>::const_iterator p2;
    std::vector< SiPixelGainCalibration::DetRegistry >::iterator p3;
    std::vector< SiPixelGainCalibration::DetRegistry >::const_iterator p4;
 
    std::vector< SiPixelGainCalibrationForHLT::DetRegistry >::iterator p5;
    std::vector< SiPixelGainCalibrationForHLT::DetRegistry >::const_iterator p6;
 
    std::vector< SiPixelGainCalibrationOffline::DetRegistry >::iterator p7;
    std::vector< SiPixelGainCalibrationOffline::DetRegistry >::const_iterator p8;
 
    std::vector<SiPixelPerformanceSummary::DetSummary>::iterator spps1;
    std::vector<SiPixelPerformanceSummary::DetSummary>::const_iterator spps2;
 
    std::vector<SiPixelQuality::disabledModuleType>::iterator p9;
    std::vector<SiPixelQuality::disabledModuleType>::const_iterator p10;
    std::vector<SiPixelDbItem> p11;
  };
}

