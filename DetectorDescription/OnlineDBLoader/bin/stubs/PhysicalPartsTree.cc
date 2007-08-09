#include <memory>

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <DetectorDescription/Core/interface/DDValue.h>
#include <DetectorDescription/Core/interface/DDsvalues.h>
#include <DetectorDescription/Core/interface/DDExpandedView.h>
#include <DetectorDescription/Core/interface/DDFilteredView.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/OfflineDBLoader/interface/ReadWriteORA.h"
#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <MagneticField/Records/interface/IdealMagneticFieldRecord.h>

#include <iostream>
#include <istream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

class PhysicalPartsTree : public edm::EDAnalyzer {

public:
 
  explicit PhysicalPartsTree( const edm::ParameterSet& );
  ~PhysicalPartsTree();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
  virtual void beginJob( const edm::EventSetup& );
  
private: 

  //std::string filename_;  
};

PhysicalPartsTree::PhysicalPartsTree( const edm::ParameterSet& iConfig )
{

}


PhysicalPartsTree::~PhysicalPartsTree()
{
}

void PhysicalPartsTree::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) { 
  std::cout << "analyze does nothing" << std::endl;
}

void PhysicalPartsTree::beginJob( const edm::EventSetup& iSetup ) {


  std::string physicalPartsTreeFileName("PHYSICALPARTSTREE.dat");
  std::string logicalPartTypesFileName("LOGICALPARTTYPES.dat");
  std::string nominalPlacementsFileName("NOMINALPLACEMENTS.dat");
  std::string detectorPartsFileName("DETECTORPARTS.dat");
  std::string pospartsGraphFileName("POSPARTSGRAPH.dat");

  std::ofstream physicalPartsTreeOS(physicalPartsTreeFileName.c_str());
  std::ofstream logicalPartTypesOS(logicalPartTypesFileName.c_str());
  std::ofstream nominalPlacementsOS(nominalPlacementsFileName.c_str());
  std::ofstream detectorPartsOS(detectorPartsFileName.c_str());
  std::ofstream pospartsGraphOS(pospartsGraphFileName.c_str());


  std::string slpname;
  std::string scategory;
  std::vector<std::string> lpname_vec;


  std::cout << "PhysicalPartsTree Analyzer..." << std::endl;
  edm::ESHandle<DDCompactView> pDD;

  iSetup.get<IdealGeometryRecord>().get( "", pDD );
  
  const DDCompactView & cpv = *pDD;
  DDExpandedView epv(cpv);
  int pospartid=0;
  while(epv.next()){
 
    //for table physicalpartstree
    std::ostringstream parent, child,logicalpartid,parentid;
    child << epv.logicalPart().name();
    const DDGeoHistory & hist = epv.geoHistory();
    parent << hist[hist.size()-2].logicalPart().name();
    logicalpartid << epv.geoHistory();
    parentid<<hist[hist.size()-2];
    physicalPartsTreeOS<<logicalpartid.str()<<","<< parentid.str() << "," << child.str()<<std::endl;

    //for table nominalPlacements
    bool reflection = false;

    Hep3Vector xv = epv.rotation().colX();
    Hep3Vector yv = epv.rotation().colY();
    Hep3Vector zv = epv.rotation().colZ();
    if ( xv.cross(yv) * zv  < 0) {
                reflection = true;
              }
 
    nominalPlacementsOS<< logicalpartid.str()<<","
		       << epv.translation().x() << "," 
		       << epv.translation().y() << ","  
		       << epv.translation().z()<< "," 
		       << epv.rotation().xx()<< "," 
		       << epv.rotation().xy()<< "," 
		       << epv.rotation().xz()<< "," 
		       << epv.rotation().yx()<< "," 
		       << epv.rotation().yy()<< "," 
		       << epv.rotation().yz()<< "," 
		       << epv.rotation().zx()<< "," 
		       << epv.rotation().zy()<< "," 
		       << epv.rotation().zz()<< "," 
		       << (int)reflection
		       <<std::endl;


    //for table DetectorParts
    detectorPartsOS <<epv.logicalPart().solid().toString() << ","
		    <<epv.logicalPart().material().toString() << ","
		    <<logicalpartid.str()
		    <<std::endl;
   
    //for table PosPartsGraph 

    pospartsGraphOS<<++pospartid<< ","<<epv.copyno()<< ","
    <<parent.str() << "," << child.str()<<std::endl;


    //for table logicalPartTypes
    //this table is special because the DPNAMEs are subset of Physicalpart_id
    //while we're looping all the Physicalpart_ids in epv, so we just 
    //keep the unique DPNAMEs into a vector and save them in looping epv.
    // if there's no lpname in lpname_vec
    //>> the OS file.
 
    slpname=epv.logicalPart().toString();

    std::vector<std::string>::iterator it = find(lpname_vec.begin(), lpname_vec.end(), slpname);    
   
    if (it==lpname_vec.end()){
      if(DDEnums::categoryName(epv.logicalPart().category())){     
	std::ostringstream category;
	category<<DDEnums::categoryName(epv.logicalPart().category());
	scategory=category.str();
	std::string unspec="unspecified";
	if(scategory==unspec){
	  lpname_vec.push_back(slpname);
	  logicalPartTypesOS <<slpname<< ","
			     <<"DETECTORPARTS"<<std::endl;
 	}
	else{
	  lpname_vec.push_back(slpname);
	  logicalPartTypesOS <<slpname<< ","<<scategory<<std::endl;

	}
      }
      else{
	lpname_vec.push_back(slpname);
	logicalPartTypesOS <<slpname<< ","
			   <<"DETECTORPARTS"<<std::endl;

      }
    }



  } //finish looping through expanded view.
 

  physicalPartsTreeOS.close();
  logicalPartTypesOS.close();
  nominalPlacementsOS.close();
  detectorPartsOS.close();
  pospartsGraphOS.close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(PhysicalPartsTree);
