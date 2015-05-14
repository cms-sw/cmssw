#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
#include "CondFormats/DataRecord/interface/SiPixelDynamicInefficiencyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CondTools/SiPixel/test/SiPixelDynamicInefficiencyReader.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>


using namespace cms;

SiPixelDynamicInefficiencyReader::SiPixelDynamicInefficiencyReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false))
{
}

SiPixelDynamicInefficiencyReader::~SiPixelDynamicInefficiencyReader(){}

void SiPixelDynamicInefficiencyReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){
  edm::ESHandle<SiPixelDynamicInefficiency> SiPixelDynamicInefficiency_; 
  iSetup.get<SiPixelDynamicInefficiencyRcd>().get(SiPixelDynamicInefficiency_);
  edm::LogInfo("SiPixelDynamicInefficiencyReader") << "[SiPixelDynamicInefficiencyReader::analyze] End Reading SiPixelDynamicInefficiency" << std::endl;

  std::map<unsigned int,double> map_pixelgeomfactor = SiPixelDynamicInefficiency_->getPixelGeomFactors();
  std::map<unsigned int,double> map_colgeomfactor = SiPixelDynamicInefficiency_->getColGeomFactors();
  std::map<unsigned int,double> map_chipgeomfactor = SiPixelDynamicInefficiency_->getChipGeomFactors();
  std::map<unsigned int,std::vector<double> > map_pufactor = SiPixelDynamicInefficiency_->getPUFactors();
  std::map<unsigned int,double>::const_iterator it_pixelgeom;
  std::map<unsigned int,double>::const_iterator it_colgeom;
  std::map<unsigned int,double>::const_iterator it_chipgeom;
  std::map<unsigned int,std::vector<double> >::const_iterator it_pu;

  std::cout<<"Printing out DB content:"<<std::endl;
  for (it_pixelgeom=map_pixelgeomfactor.begin();it_pixelgeom!=map_pixelgeomfactor.end();it_pixelgeom++)
  {
    printf("pixelgeom detid %x\tfactor %f\n",it_pixelgeom->first,it_pixelgeom->second);
  }
  for (it_colgeom=map_colgeomfactor.begin();it_colgeom!=map_colgeomfactor.end();it_colgeom++)
  {
    printf("colgeom detid %x\tfactor %f\n",it_colgeom->first,it_colgeom->second);
  }
  for (it_chipgeom=map_chipgeomfactor.begin();it_chipgeom!=map_chipgeomfactor.end();it_chipgeom++)
  {
    printf("chipgeom detid %x\tfactor %f\n",it_chipgeom->first,it_chipgeom->second);
  }
  for (it_pu=map_pufactor.begin();it_pu!=map_pufactor.end();it_pu++)
  {
    printf("pu detid %x\t",it_pu->first);
    std::cout  << " Size of vector "<<it_pu->second.size()<<" elements:";
    if (it_pu->second.size()>1) {
      for (unsigned int i=0;i<it_pu->second.size();i++) {
        std::cout<<" "<<it_pu->second.at(i);
      }
      std::cout<<std::endl;
    }
    else {
      std::cout<<" "<<it_pu->second.at(0)<<std::endl;
    }
  }
  std::vector<uint32_t> detIdmasks = SiPixelDynamicInefficiency_->getDetIdmasks();
  for (unsigned int i=0;i<detIdmasks.size();i++) printf("DetId Mask: %x\t\n",detIdmasks.at(i));
  double theInstLumiScaleFactor = SiPixelDynamicInefficiency_->gettheInstLumiScaleFactor_();
  std::cout<<"theInstLumiScaleFactor "<<theInstLumiScaleFactor<<std::endl;

}
