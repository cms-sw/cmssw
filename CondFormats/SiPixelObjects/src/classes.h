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
