#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
namespace {
  std::map<SiPixelFedCablingMap::Key, sipixelobjects::PixelROC> theMap; 
  std::pair<const SiPixelFedCablingMap::Key, sipixelobjects::PixelROC> theMapValueT; 
}

#include "CondFormats/SiPixelObjects/interface/SiPixelCPEGenericErrorParm.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPedestals.h"
namespace{
  std::map< unsigned int, SiPixelPedestals::SiPixelPedestalsVector> sipixped;
}

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
namespace { 
  std::vector<char>::iterator p1;
  std::vector<char>::const_iterator p2;
  std::vector< SiPixelGainCalibration::DetRegistry >::iterator p3;
  std::vector< SiPixelGainCalibration::DetRegistry >::const_iterator p4;
}

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
namespace {
  std::vector< SiPixelGainCalibrationForHLT::DetRegistry >::iterator p5;
  std::vector< SiPixelGainCalibrationForHLT::DetRegistry >::const_iterator p6;
}

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationOffline.h"
namespace {
  std::vector< SiPixelGainCalibrationOffline::DetRegistry >::iterator p7;
  std::vector< SiPixelGainCalibrationOffline::DetRegistry >::const_iterator p8;
}

#include "CondFormats/SiPixelObjects/interface/SiPixelTemplate.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelTemplateDBObject.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelPerformanceSummary.h"
namespace{
  std::vector<SiPixelPerformanceSummary::DetSummary>::iterator spps1;
  std::vector<SiPixelPerformanceSummary::DetSummary>::const_iterator spps2;
}  

#include "CondFormats/SiPixelObjects/interface/SiPixelCalibConfiguration.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
namespace{
  std::vector<SiPixelQuality::disabledModuleType>::iterator p9;
  std::vector<SiPixelQuality::disabledModuleType>::const_iterator p10;
}

#include "CondFormats/SiPixelObjects/interface/PixelDCSObject.h"
template class PixelDCSObject<bool>;
template class PixelDCSObject<float>;
template class PixelDCSObject<CaenChannel>;
