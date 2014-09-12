/// DQM and Framework services
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizerPhase1.h"
//#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
/// Data Formats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
#include <sstream>
#include <cstdio>

/// Constructor
SiPixelFolderOrganizerPhase1::SiPixelFolderOrganizerPhase1(bool getStore) :
  rootFolder("Pixel")
{  
  //Not allowed in multithread framework, but can still be called by other modules not from DQM.
  if (getStore) dbe_ = edm::Service<DQMStore>().operator->();
}

SiPixelFolderOrganizerPhase1::~SiPixelFolderOrganizerPhase1() {}

//Overloaded function for calling outside of DQM framework
bool SiPixelFolderOrganizerPhase1::setModuleFolder(const uint32_t& rawdetid, int type) {

  bool flag = false;

   if(rawdetid == 0) {
     dbe_->setCurrentFolder(rootFolder);
     flag = true;
   }
   ///
   /// Pixel Barrel
   ///
   else if(DetId(rawdetid).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {

     //for endcap types there is nothing to do: 
     if(type>3 && type!=7) return true;
     
     std::string subDetectorFolder = "Barrel";
     PixelBarrelNameUpgrade::Shell DBshell = PixelBarrelNameUpgrade(DetId(rawdetid)).shell();
     int DBlayer  = PixelBarrelNameUpgrade(DetId(rawdetid)).layerName();
     int DBladder = PixelBarrelNameUpgrade(DetId(rawdetid)).ladderName();
     int DBmodule = PixelBarrelNameUpgrade(DetId(rawdetid)).moduleName();
     
     char slayer[80];  sprintf(slayer, "Layer_%i",   DBlayer);
     char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
     char smodule[80]; sprintf(smodule,"Module_%i",  DBmodule);
     
     std::ostringstream sfolder;
     
     sfolder << rootFolder << "/" << subDetectorFolder; 
     if(type<4){
       sfolder << "/Shell_" <<DBshell
               << "/" << slayer;
     } 
     if(type<2){
       sfolder << "/" << sladder;
       if ( PixelBarrelNameUpgrade(DetId(rawdetid)).isHalfModule() ) sfolder <<"H"; 
       else sfolder <<"F";
     }
     if(type==0) sfolder << "/" <<smodule;
     //if(type==3) sfolder << "/all_" << smodule;
     
     //std::cout<<"set barrel folder: "<<rawdetid<<" : "<<sfolder.str().c_str()<<std::endl;
     
     dbe_->setCurrentFolder(sfolder.str().c_str());
     flag = true;
   } 
   
   ///
   /// Pixel Endcap
   ///
   else if(DetId(rawdetid).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {

     //for barrel types there is nothing to do: 
     if(type>0 && type < 4) return true;
     
     std::string subDetectorFolder = "Endcap";
     PixelEndcapNameUpgrade::HalfCylinder side = PixelEndcapNameUpgrade(DetId(rawdetid)).halfCylinder();
     int disk   = PixelEndcapNameUpgrade(DetId(rawdetid)).diskName();
     int blade  = PixelEndcapNameUpgrade(DetId(rawdetid)).bladeName();
     int panel  = PixelEndcapNameUpgrade(DetId(rawdetid)).pannelName();
     int module = PixelEndcapNameUpgrade(DetId(rawdetid)).plaquetteName();
     
     
     char sdisk[80];  sprintf(sdisk,  "Disk_%i",disk);
     char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
     char spanel[80]; sprintf(spanel, "Panel_%i",panel);
     char smodule[80];sprintf(smodule,"Module_%i",module);
     
     std::ostringstream sfolder;
     
     sfolder <<rootFolder <<"/" << subDetectorFolder << 
       "/HalfCylinder_" << side << "/" << sdisk; 
     if(type==0 || type ==4){
       sfolder << "/" << sblade; 
     }
     if(type==0){
       sfolder << "/" << spanel << "/" << smodule;
     }
     //        if(type==6){
     //          sfolder << "/" << spanel << "_all_" << smodule;
     //        }
     
     //std::cout<<"set endcap folder: "<<rawdetid<<" : "<<sfolder.str().c_str()<<std::endl;
     
     dbe_->setCurrentFolder(sfolder.str().c_str());
     flag = true;
   } else throw cms::Exception("LogicError")
	    << "[SiPixelFolderOrganizerPhase1::setModuleFolder] Not a Pixel detector DetId ";
   
   return flag;

}

//Overloaded setModuleFolder for multithread safe operation
bool SiPixelFolderOrganizerPhase1::setModuleFolder(DQMStore::IBooker& iBooker, const uint32_t& rawdetid, int type) {

  bool flag = false;

   if(rawdetid == 0) {
     iBooker.setCurrentFolder(rootFolder);
     flag = true;
   }
   ///
   /// Pixel Barrel
   ///
   else if(DetId(rawdetid).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {

     //for endcap types there is nothing to do: 
     if(type>3 && type!=7) return true;
     
     std::string subDetectorFolder = "Barrel";
     PixelBarrelNameUpgrade::Shell DBshell = PixelBarrelNameUpgrade(DetId(rawdetid)).shell();
     int DBlayer  = PixelBarrelNameUpgrade(DetId(rawdetid)).layerName();
     int DBladder = PixelBarrelNameUpgrade(DetId(rawdetid)).ladderName();
     int DBmodule = PixelBarrelNameUpgrade(DetId(rawdetid)).moduleName();
     
     char slayer[80];  sprintf(slayer, "Layer_%i",   DBlayer);
     char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
     char smodule[80]; sprintf(smodule,"Module_%i",  DBmodule);
     
     std::ostringstream sfolder;
     
     sfolder << rootFolder << "/" << subDetectorFolder; 
     if(type<4){
       sfolder << "/Shell_" <<DBshell
               << "/" << slayer;
     } 
     if(type<2){
       sfolder << "/" << sladder;
       if ( PixelBarrelNameUpgrade(DetId(rawdetid)).isHalfModule() ) sfolder <<"H"; 
       else sfolder <<"F";
     }
     if(type==0) sfolder << "/" <<smodule;
     //if(type==3) sfolder << "/all_" << smodule;
     
     //std::cout<<"set barrel folder: "<<rawdetid<<" : "<<sfolder.str().c_str()<<std::endl;
     
     iBooker.setCurrentFolder(sfolder.str().c_str());
     flag = true;
   } 
   
   ///
   /// Pixel Endcap
   ///
   else if(DetId(rawdetid).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {

     //for barrel types there is nothing to do: 
     if(type>0 && type < 4) return true;
     
     std::string subDetectorFolder = "Endcap";
     PixelEndcapNameUpgrade::HalfCylinder side = PixelEndcapNameUpgrade(DetId(rawdetid)).halfCylinder();
     int disk   = PixelEndcapNameUpgrade(DetId(rawdetid)).diskName();
     int blade  = PixelEndcapNameUpgrade(DetId(rawdetid)).bladeName();
     int panel  = PixelEndcapNameUpgrade(DetId(rawdetid)).pannelName();
     int module = PixelEndcapNameUpgrade(DetId(rawdetid)).plaquetteName();
     
     
     char sdisk[80];  sprintf(sdisk,  "Disk_%i",disk);
     char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
     char spanel[80]; sprintf(spanel, "Panel_%i",panel);
     char smodule[80];sprintf(smodule,"Module_%i",module);
     
     std::ostringstream sfolder;
     
     sfolder <<rootFolder <<"/" << subDetectorFolder << 
       "/HalfCylinder_" << side << "/" << sdisk; 
     if(type==0 || type ==4){
       sfolder << "/" << sblade; 
     }
     if(type==0){
       sfolder << "/" << spanel << "/" << smodule;
     }
     //        if(type==6){
     //          sfolder << "/" << spanel << "_all_" << smodule;
     //        }
     
     //std::cout<<"set endcap folder: "<<rawdetid<<" : "<<sfolder.str().c_str()<<std::endl;
     
     iBooker.setCurrentFolder(sfolder.str().c_str());
     flag = true;
   } else throw cms::Exception("LogicError")
	    << "[SiPixelFolderOrganizerPhase1::setModuleFolder] Not a Pixel detector DetId ";
   
   return flag;

}

//Overloaded setFedFolder for use outside of DQM framework
bool SiPixelFolderOrganizerPhase1::setFedFolder(const uint32_t FedId) {

  std::string subDetectorFolder = "AdditionalPixelErrors";
  char sFed[80];  sprintf(sFed,  "FED_%i",FedId);
  std::ostringstream sfolder;
  
  sfolder << rootFolder << "/" << subDetectorFolder << "/" << sFed;
  dbe_->setCurrentFolder(sfolder.str().c_str());
  
  return true;

}

//Overloaded setFedFolder to avoid accessing DQMStore directly.
bool SiPixelFolderOrganizerPhase1::setFedFolder(DQMStore::IBooker& iBooker, const uint32_t FedId) {

  std::string subDetectorFolder = "AdditionalPixelErrors";
  char sFed[80];  sprintf(sFed,  "FED_%i",FedId);
  std::ostringstream sfolder;
  
  sfolder << rootFolder << "/" << subDetectorFolder << "/" << sFed;
  iBooker.setCurrentFolder(sfolder.str().c_str());
  
  return true;

}

void SiPixelFolderOrganizerPhase1::getModuleFolder(const uint32_t& rawdetid, 
                                             std::string& path) {

  path = rootFolder;
  if(rawdetid == 0) {
    return;
  }else if( (DetId(rawdetid).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) ) {
    std::string subDetectorFolder = "Barrel";
    PixelBarrelNameUpgrade::Shell DBshell = PixelBarrelNameUpgrade(DetId(rawdetid)).shell();
    int DBlayer  = PixelBarrelNameUpgrade(DetId(rawdetid)).layerName();
    int DBladder = PixelBarrelNameUpgrade(DetId(rawdetid)).ladderName();
    int DBmodule = PixelBarrelNameUpgrade(DetId(rawdetid)).moduleName();
    
    //char sshell[80];  sprintf(sshell, "Shell_%i",   DBshell);
    char slayer[80];  sprintf(slayer, "Layer_%i",   DBlayer);
    char sladder[80]; sprintf(sladder,"Ladder_%02i",DBladder);
    char smodule[80]; sprintf(smodule,"Module_%i",  DBmodule);
    
    std::ostringstream sfolder;
    sfolder << rootFolder << "/" << subDetectorFolder << "/Shell_" <<DBshell << "/" << slayer << "/" << sladder;
    if ( PixelBarrelNameUpgrade(DetId(rawdetid)).isHalfModule() ) sfolder <<"H"; 
    else sfolder <<"F";
    sfolder << "/" <<smodule;
    path = sfolder.str().c_str();
   
    //path = path + "/" + subDetectorFolder + "/" + sshell + "/" + slayer + "/" + sladder;
    //if(PixelBarrelNameUpgrade(DetId(rawdetid)).isHalfModule() )
    //  path = path + "H"; 
    //else path = path + "F";
    //path = path + "/" + smodule;

  } else if( (DetId(rawdetid).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) ) {
    std::string subDetectorFolder = "Endcap";
    PixelEndcapNameUpgrade::HalfCylinder side = PixelEndcapNameUpgrade(DetId(rawdetid)).halfCylinder();
    int disk   = PixelEndcapNameUpgrade(DetId(rawdetid)).diskName();
    int blade  = PixelEndcapNameUpgrade(DetId(rawdetid)).bladeName();
    int panel  = PixelEndcapNameUpgrade(DetId(rawdetid)).pannelName();
    int module = PixelEndcapNameUpgrade(DetId(rawdetid)).plaquetteName();

    //char shc[80];  sprintf(shc,  "HalfCylinder_%i",side);
    char sdisk[80];  sprintf(sdisk,  "Disk_%i",disk);
    char sblade[80]; sprintf(sblade, "Blade_%02i",blade);
    char spanel[80]; sprintf(spanel, "Panel_%i",panel);
    char smodule[80];sprintf(smodule,"Module_%i",module);

    std::ostringstream sfolder;
    sfolder <<rootFolder <<"/" << subDetectorFolder << "/HalfCylinder_" << side << "/" << sdisk << "/" << sblade << "/" << spanel << "/" << smodule;
    path = sfolder.str().c_str();
    
    //path = path + "/" + subDetectorFolder + "/" + shc + "/" + sdisk + "/" + sblade + "/" + spanel + "/" + smodule;

  }else throw cms::Exception("LogicError")
     << "[SiPixelFolderOrganizerPhase1::getModuleFolder] Not a Pixel detector DetId ";
     
  //std::cout<<"resulting final path name: "<<path<<std::endl;   
     
  return;
}
