#include "CondFormats/SiPixelObjects/interface/SiPixelPedestals.h"
namespace{
  std::map< unsigned int, SiPixelPedestals::SiPixelPedestalsVector> sipixped;
}

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
namespace{
  std::vector<sipixelobjects::PixelROC> theROCs;
//  std::vector<sipixelobjects::PixelFEDLink::Connection> theConnections;
}

#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
namespace{
  std::vector<sipixelobjects::PixelFEDLink> theLinks;
}

#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
namespace{
  std::vector<sipixelobjects::PixelFEDCabling> theFedCablings;
}

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
std::vector<char>::iterator p1;
std::vector<char>::const_iterator p2;
std::vector< SiPixelGainCalibration::DetRegistry >::iterator p3;
std::vector< SiPixelGainCalibration::DetRegistry >::const_iterator p4;

#include "CondFormats/SiPixelObjects/interface/SiPixelTemplate.h"
// &&& Not sure what we need for templates here.

#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"