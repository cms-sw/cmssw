/// DQM and Framework services
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
//#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
/// Data Formats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

/// Constructor
SiPixelFolderOrganizer::SiPixelFolderOrganizer() :
  rootFolder("Tracker"),
  slash("/"),
  dbe_(edm::Service<DaqMonitorBEInterface>().operator->())
{  
}

SiPixelFolderOrganizer::~SiPixelFolderOrganizer() {}

bool SiPixelFolderOrganizer::setModuleFolder(const uint32_t& rawdetid) {

  bool flag = false;

   if(rawdetid == 0) {
     dbe_->setCurrentFolder(rootFolder);
     flag = true;
   }
   ///
   /// Pixel Barrel
   ///
   else if(DetId::DetId(rawdetid).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
     
     std::string subDetectorFolder = "PixelBarrel";
     PixelBarrelName::Shell DBshell = PixelBarrelName::PixelBarrelName(DetId::DetId(rawdetid)).shell();
     int DBlayer  = PixelBarrelName::PixelBarrelName(DetId::DetId(rawdetid)).layerName();
     int DBladder = PixelBarrelName::PixelBarrelName(DetId::DetId(rawdetid)).ladderName();
     int DBmodule = PixelBarrelName::PixelBarrelName(DetId::DetId(rawdetid)).moduleName();
     
     char slayer[80];  sprintf(slayer, "Layer_%i",   DBlayer);
     char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
     char smodule[80]; sprintf(smodule,"Module_%i",  DBmodule);
     
     std::ostringstream sfolder;
     
     sfolder << rootFolder << "/" << subDetectorFolder << "/Shell_" <<DBshell<< "/" << slayer << "/" << sladder;
     if ( PixelBarrelName::PixelBarrelName(DetId::DetId(rawdetid)).isHalfModule() ) sfolder <<"H"; 
     else sfolder <<"F";
     sfolder << "/" <<smodule;
     
     dbe_->setCurrentFolder(sfolder.str().c_str());
     flag = true;
   } 
   ///
   /// Pixel Endcap
   ///
   else if(DetId::DetId(rawdetid).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {

     std::string subDetectorFolder = "PixelEndcap";
     PixelEndcapName::HalfCylinder side = PixelEndcapName::PixelEndcapName(DetId::DetId(rawdetid)).halfCylinder();
      int disk   = PixelEndcapName::PixelEndcapName(DetId::DetId(rawdetid)).diskName();
      int blade  = PixelEndcapName::PixelEndcapName(DetId::DetId(rawdetid)).bladeName();
      int panel  = PixelEndcapName::PixelEndcapName(DetId::DetId(rawdetid)).pannelName();
      int module = PixelEndcapName::PixelEndcapName(DetId::DetId(rawdetid)).plaquetteName();

      char sdisk[80];  sprintf(sdisk,  "Disk_%i",disk);
      char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
      char spanel[80]; sprintf(spanel, "Panel_%i",panel);
      char smodule[80];sprintf(smodule,"Module_%i",module);

      std::ostringstream sfolder;

      sfolder <<rootFolder <<"/" << subDetectorFolder << 
	"/HalfCylinder_" << side << "/" << sdisk << 
	"/" << sblade << "/" << spanel << "/" << smodule;

      dbe_->setCurrentFolder(sfolder.str().c_str());
      flag = true;

   } else throw cms::Exception("LogicError")
     << "[SiPixelFolderOrganizer::setModuleFolder] Not a Pixel detector DetId ";
   
   return flag;

}

bool SiPixelFolderOrganizer::setFedFolder(const uint32_t FedId) {

  std::string subDetectorFolder = "AdditionalPixelErrors";
  char sFed[80];  sprintf(sFed,  "FED_%i",FedId);
  std::ostringstream sfolder;
  
  sfolder << rootFolder << "/" << subDetectorFolder << "/" << sFed;
  dbe_->setCurrentFolder(sfolder.str().c_str());
  
  return true;

}
