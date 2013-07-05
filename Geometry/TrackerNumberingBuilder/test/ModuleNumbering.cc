// -*- C++ -*-
//
/* 
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

//
// Original Author:  Riccardo Ranieri
//         Created:  Tue Feb 27 22:22:22 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

// output
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <bitset>
//

//
// class decleration
//

//double PI = 3.141592654;

class ModuleNumbering : public edm::EDAnalyzer {
public:
  explicit ModuleNumbering( const edm::ParameterSet& );
  ~ModuleNumbering();
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  void fillModuleVariables(const GeometricDet* module, double& polarRadius, double& phiRad, double& z);
  double changePhiRange_From_ZeroTwoPi_To_MinusPiPlusPi(double phiRad);
  double changePhiRange_From_MinusPiPlusPi_To_MinusTwoPiZero(double phiRad);
  double changePhiRange_From_MinusPiPlusPi_To_ZeroTwoPi(double phiRad);
  //
  // counters
  unsigned int iOK;
  unsigned int iERROR;
  //
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
static const double tolerance_space = 1.000; // 1.000 mm
static const double tolerance_angle = 0.001; // 0.001 rad

//
// constructors and destructor
//
ModuleNumbering::ModuleNumbering( const edm::ParameterSet& iConfig )
{
  //now do what ever initialization is needed

}


ModuleNumbering::~ModuleNumbering()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//

void ModuleNumbering::fillModuleVariables(const GeometricDet* module, double& polarRadius, double& phiRad, double& z) {
  // module variables
  polarRadius = std::sqrt(module->translation().X()*module->translation().X()+module->translation().Y()*module->translation().Y());
  phiRad = atan2(module->translation().Y(),module->translation().X());
  // tolerance near phi=0
  if(fabs(phiRad) < tolerance_angle) phiRad=0.0;
  // negative phi: from [-PI,+PI) to [0,2PI)
  if(phiRad < 0) phiRad+=2*M_PI;
  //
  z = module->translation().Z();
  //
}

double ModuleNumbering::changePhiRange_From_ZeroTwoPi_To_MinusPiPlusPi(double phiRad) {
  double new_phiRad = phiRad;
  // tolerance near phi=PI
  if(fabs(new_phiRad-M_PI) < tolerance_angle) new_phiRad=M_PI;
  // phi greater than PI: from [0,2PI) to [-PI,+PI)
  if(new_phiRad > M_PI) new_phiRad-=2*M_PI;
  //
  return new_phiRad;
}

double ModuleNumbering::changePhiRange_From_MinusPiPlusPi_To_MinusTwoPiZero(double phiRad) {
  double new_phiRad = phiRad;
  // tolerance near phi=PI
  if(fabs(fabs(new_phiRad)-M_PI) < tolerance_angle) new_phiRad=M_PI;
  // phi greater than PI: from [-PI,+PI) to [0,2PI)
  if(new_phiRad > 0) new_phiRad-=2*M_PI;
  //
  return new_phiRad;
}

double ModuleNumbering::changePhiRange_From_MinusPiPlusPi_To_ZeroTwoPi(double phiRad) {
  double new_phiRad = phiRad;
  // tolerance near phi=PI
  if(fabs(fabs(new_phiRad)-M_PI) < tolerance_angle) new_phiRad=M_PI;
  // phi greater than PI: from [-PI,+PI) to [0,2PI)
  if(new_phiRad < 0) new_phiRad+=2*M_PI;
  //
  return new_phiRad;
}

// ------------ method called to produce the data  ------------
void
ModuleNumbering::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  iSetup.get<IdealGeometryRecord>().get(tTopo);


  
  edm::LogInfo("ModuleNumbering") << "begins";
  
  // output file
  std::ofstream Output("ModuleNumbering.log",std::ios::out);
  //
  
  // reset counters
  iOK    = 0;
  iERROR = 0;
  //
  
  //
  // get the GeometricDet
  //
  edm::ESHandle<GeometricDet> rDD;
  edm::ESHandle<std::vector<GeometricDetExtra> > rDDE;
  iSetup.get<IdealGeometryRecord>().get( rDD );     
  iSetup.get<IdealGeometryRecord>().get( rDDE );     
  edm::LogInfo("ModuleNumbering") << " Top node is  " << rDD.product() << " " <<  rDD.product()->name().name() << std::endl;
  edm::LogInfo("ModuleNumbering") << " And Contains  Daughters: " << rDD.product()->deepComponents().size() << std::endl;
  CmsTrackerDebugNavigator nav(*rDDE.product());
  nav.dump(*rDD.product(), *rDDE.product());
  //
  //first instance tracking geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
  //
 
  std::vector<const GeometricDet*> modules =  (*rDD).deepComponents();
  std::map< uint32_t , const GeometricDet* > mapDetIdToGeometricDet;
  
  for(unsigned int i=0; i<modules.size();i++){  
    mapDetIdToGeometricDet[modules[i]->geographicalID().rawId()] = modules[i];
  }
  
  // Debug variables
  //
  uint32_t     myDetId       = 0;
  unsigned int iDetector     = 1;
  unsigned int nSubDetectors = 6;
  //
  double polarRadius = 0.0;
  double phiRad      = 0.0;
  double z           = 0.0;
  //
  
  Output << "************************ List of modules with positions ************************" << std::endl;
  Output << std::fixed << std::setprecision(4); // set as default 4 decimal digits (0.1 um or 0.1 rad)
  
  for(unsigned int iSubDetector = 1; iSubDetector <= nSubDetectors; iSubDetector++) {
    
    // modules
    switch (iSubDetector) {
      
      // PXB
    case 1:
      {
	break;
      }
      
      // PXF
    case 2:
      {
	break;
      }
      
      // TIB
    case 3:
      {
	
	// TIB loop
	unsigned int string_int_ext_TIB[8] = { 26 , 30 , 34 , 38 , 44 , 46 , 52 , 56 }; // number of strings per layer 1/4 int/ext
	unsigned int mod_type_TIB[8] = { 1 , 2 , 1 , 2 , 0 , 0 , 0 , 0 }; // first and last type for module type loop
	unsigned int nLayers  = 4;
	unsigned int nModules = 3;
	// debug variables
	double layer_R             = 0.0;
	double layer_R_previous    = 0.0;
	double side_z              = 0.0;
	double side_z_previous     = -10000.0;
	double part_R              = 0.0;
	double part_R_previous     = 0.0;
	double string_phi          = 0.0;
	double string_phi_previous = 0.0;
	double module_z            = 0.0;
	double module_z_previous   = 0.0;
	//
	for(unsigned int iLayer = 1; iLayer <= nLayers; iLayer++) { // Layer: 1,...,nLayers
	  for(unsigned int iSide = 1; iSide <= 2; iSide++){ // Side: 1:- 2:+
	    for(unsigned int iPart = 1; iPart <= 2; iPart++){ // Part: 1:int 2:ext
	      unsigned int jString = ( 2 * ( iLayer - 1 ) ) + ( iPart - 1 );
	      for(unsigned int iString = 1; iString <= string_int_ext_TIB[jString] ; iString++) {  // String: 1,...,nStrings
		for(unsigned int iModule = 1; iModule <= nModules ; iModule++) {  // Module: 1,...,nModules
		  for(unsigned int iType = mod_type_TIB[2*(iLayer-1)]; iType <= mod_type_TIB[2*(iLayer-1)+1] ; iType++) { // Module Type: 0 (ss) 1-2 (ds stereo and rphi)
		    
		  myDetId = 0;
		  // detector
		  myDetId <<= 4;
		  myDetId |= iDetector;
		  // subdetector
		  myDetId <<= 3;
		  myDetId |= iSubDetector;
		  // not used
		  myDetId <<= 8;
		  myDetId |= 0;
		  // layer
		  myDetId <<= 3;
		  myDetId |= iLayer;
		  // side
		  myDetId <<= 2;
		  myDetId |= iSide;
		  // part
		  myDetId <<= 2;
		  myDetId |= iPart;
		  // string number
		  myDetId <<= 6;
		  myDetId |= iString;
		  // module number
		  myDetId <<= 2;
		  myDetId |= iModule;
		  // module type
		  myDetId <<= 2;
		  myDetId |= iType;
		  //
		  std::bitset<32> binary_myDetId(myDetId);
		  Output << std::endl  << std::endl;
		  Output << " ******** myDet Id = " << myDetId << " (" << binary_myDetId << ")" << std::endl;
		  //
		  unsigned int rawid = mapDetIdToGeometricDet[myDetId]->geographicalID().rawId();
		  std::bitset<32> binary_detid(rawid);
		  GeometricDet::nav_type detNavType = mapDetIdToGeometricDet[myDetId]->navType();
		  //
		  Output << "            raw Id = " << rawid << " (" << binary_detid << ")"
			 << "\t nav type = " << printNavType(Output,&detNavType.front(),detNavType.size()) << std::endl;
		  
		  // variables
		  fillModuleVariables(mapDetIdToGeometricDet[myDetId], polarRadius, phiRad, z);
		  layer_R    = polarRadius;
		  side_z     = z;
		  part_R     = polarRadius;
		  string_phi = phiRad;
		  module_z   = z;
		  //
		  
		  Output << "\t R = " << polarRadius << "\t phi = " << phiRad << "\t z = " << z << std::endl;
		  
		  // Module Info
		  
		  std::string name = mapDetIdToGeometricDet[myDetId]->name().name();
		  unsigned int         theLayer  = tTopo->tibLayer(rawid);
		  std::vector<unsigned int> theString = tTopo->tibStringInfo(rawid);
		  unsigned int         theModule = tTopo->tibModule(rawid);
		  std::string side;
		  std::string part;
		  side = (theString[0] == 1 ) ? "-" : "+";
		  part = (theString[1] == 1 ) ? "int" : "ext";
		  Output << " TIB" << side << "\t" << "Layer " << theLayer << " " << part
			 << "\t" << "string " << theString[2] << "\t" << " module " << theModule << " " << name << std::endl;
		  //
		  
		  // module: |z| check
		  Output << "\t # ";
		  if( ( fabs(module_z) - fabs(module_z_previous) ) < (0 + tolerance_space) ) {
		    Output << "\t ERROR |z| ordering not respected in module ";
		    iERROR++;
		  } else {
		    Output << "\t OK" << " |z| ordering in module ";
		    iOK++;
		  }
		  Output << iModule-1 << " to " << iModule << " (" << module_z_previous << " --> " << module_z << ")" << std::endl;
		  //
		  } // type loop
		  
		  //
		  module_z_previous = module_z;
		  //
		} // module loop
		
	        // string: phi check
		Output << "\t ## ";
		if( ( string_phi - string_phi_previous ) < (0 - tolerance_angle) ) {
		  Output << "\t ERROR phi ordering not respected in string ";
		  iERROR++;
		} else {
		  Output << "\t OK" << " phi ordering in string ";
		  iOK++;
		}
		Output << iString-1 << " to " << iString << " (" << string_phi_previous << " --> " << string_phi << ")" << std::endl;
		//
		string_phi_previous = string_phi;
		module_z_previous   = 0.0;
		//		
	      } // string loop
	      
	      // part: R check
	      Output << "\t ### ";
	      if( ( part_R - part_R_previous ) < (0 + tolerance_space) ) {
		Output << "\t ERROR R ordering (part int/ext) not respected in layer ";
		iERROR++;
	      } else {
		Output << "\t OK" << " R ordering (part int/ext) in layer ";
		iOK++;
	      }
	      Output << iLayer << " part " << iPart-1 << " to " << iPart << " (" << part_R_previous << " --> " << part_R << ")" << std::endl;
	      //
	      part_R_previous     = part_R;
	      string_phi_previous = 0.0;
	      module_z_previous   = 0.0;
	      //
	    } // part loop
	    
	    // side: z check
	    Output << "\t #### ";
	    if( ( side_z - side_z_previous ) < (0 + tolerance_space) ) {
	      Output << "\t ERROR z ordering (side -/+) not respected in layer ";
	      iERROR++;
	    } else {
	      Output << "\t OK" << " z ordering (side -/+) in layer ";
	      iOK++;
	    }
	    Output << iLayer << " side " << iSide-1 << " to " << iSide << " (" << side_z_previous << " --> " << side_z << ")" << std::endl;
	    //
	    side_z_previous     = side_z;
	    part_R_previous     = 0.0;
	    string_phi_previous = 0.0;
	    module_z_previous   = 0.0;
	  //	    
	  } // side loop
	  
	  // layer: R check
	  Output << "\t ##### ";
	  if( ( layer_R - layer_R_previous ) < (0 + tolerance_space) ) {
	    Output << "\t ERROR R ordering not respected from layer ";
	    iERROR++;
	  } else {
	    Output << "\t OK" << " R ordering in layer ";
	    iOK++;
	  }
	  Output << iLayer-1 << " to " << iLayer << " (" << layer_R_previous << " --> " << layer_R << ")" << std::endl;	    
	  //
	  layer_R_previous    = layer_R;
	  side_z_previous     = -10000.0;
	  part_R_previous     = 0.0;
	  string_phi_previous = 0.0;
	  module_z_previous   = 0.0;
	  //
	} // layer loop
	
	break;
      }
      
      // TID
    case 4:
      {

	// TID loop
	unsigned int modules_TID[3] = { 12 , 12 , 20 }; // number of modules per disk
	unsigned int mod_type_TID[8] = { 1 , 2 , 1 , 2 , 0 , 0 , 0 , 0 }; // first and last type for module type loop
	unsigned int nDisks = 3;
	unsigned int nRings = 3;
	// debug variables
	double side_z              = 0.0;
	double side_z_previous     = 10000.0;
	double disk_z              = 0.0;
	double disk_z_previous     = 0.0;
	double ring_R              = 0.0;
	double ring_R_previous     = 0.0;
	double part_z              = 0.0;
	double part_z_previous     = 0.0;
	double module_phi          = 0.0;
	double module_phi_previous = -M_PI;
	//
	for(unsigned int iSide = 2; iSide >= 1; iSide--){ // Side: 1:- 2:+
	  for(unsigned int iDisk = 1; iDisk <= nDisks; iDisk++) { // Disk: 1,...,nDisks
	    for(unsigned int iRing = 1; iRing <= nRings; iRing++){ // Ring: 1,...,nRings
	      for(int iPart = 2; iPart >= 1; iPart--){ // Part: 1:back 2:front
		for(unsigned int iModule = 1; iModule <= modules_TID[iRing-1] ; iModule++) {  // Module: 1,...,modules in ring
		  for(unsigned int iType = mod_type_TID[2*(iRing-1)]; iType <= mod_type_TID[2*(iRing-1)+1] ; iType++) { // Module Type: 0 (ss) 1-2 (ds stereo and rphi)
		    
		  myDetId = 0;
		  // detector
		  myDetId <<= 4;
		  myDetId |= iDetector;
		  // subdetector
		  myDetId <<= 3;
		  myDetId |= iSubDetector;
		  // not used
		  myDetId <<= 10;
		  myDetId |= 0;
		  // side
		  myDetId <<= 2;
		  myDetId |= iSide;
		  // disk number
		  myDetId <<= 2;
		  myDetId |= iDisk;
		  // ring number
		  myDetId <<= 2;
		  myDetId |= iRing;
		  // part
		  myDetId <<= 2;
		  myDetId |= iPart;
		  // module number
		  myDetId <<= 5;
		  myDetId |= iModule;
		  // module type
		  myDetId <<= 2;
		  myDetId |= iType;
		  //
		  std::bitset<32> binary_myDetId(myDetId);
		  Output << std::endl  << std::endl;
		  Output << " ******** myDet Id = " << myDetId << " (" << binary_myDetId << ")" << std::endl;
		  //
		  unsigned int rawid = mapDetIdToGeometricDet[myDetId]->geographicalID().rawId();
		  std::bitset<32> binary_detid(rawid);
		  GeometricDet::nav_type detNavType = mapDetIdToGeometricDet[myDetId]->navType();
		  //
		  Output << "            raw Id = " << rawid << " (" << binary_detid << ")"
			 << "\t nav type = " << printNavType(Output,&detNavType.front(),detNavType.size()) << std::endl;
		  
		  // variables
		  fillModuleVariables(mapDetIdToGeometricDet[myDetId], polarRadius, phiRad, z);
		  side_z     = z;
		  disk_z     = z;
		  ring_R     = polarRadius;
		  part_z     = z;
		  module_phi = phiRad;
		  //
		  
		  Output << "\t R = " << polarRadius << "\t phi = " << phiRad << "\t z = " << z << std::endl;
		  
		  // Module Info
		  
		  std::string name = mapDetIdToGeometricDet[myDetId]->name().name();
		  unsigned int         theDisk   = tTopo->tidWheel(rawid);
		  unsigned int         theRing   = tTopo->tidRing(rawid);
		  std::vector<unsigned int> theModule = tTopo->tidModuleInfo(rawid);
		  std::string side;
		  std::string part;
		  side = (tTopo->tidSide(rawid) == 1 ) ? "-" : "+";
		  part = (theModule[0] == 1 ) ? "back" : "front";
		  Output << " TID" << side << "\t" << "Disk " << theDisk << " Ring " << theRing << " " << part
			 << "\t" << " module " << theModule[1] << "\t" << name << std::endl;
		  //
		  
		  // module: phi check
		  Output << "\t # ";
		  if( ( module_phi - module_phi_previous ) < (0 - tolerance_angle) ) {
		    Output << "\t ERROR phi ordering not respected in module ";
		    iERROR++;
		  } else {
		    Output << "\t OK" << " phi ordering in module ";
		    iOK++;
		  }
		  Output << iModule-1 << " to " << iModule << " (" << module_phi_previous << " --> " << module_phi << ")" << std::endl;
		  //
		  } // type loop
		  
		  //
		  module_phi_previous = module_phi;
		  //
		} // module loop
		
		// part: |z| check
		Output << "\t ## ";
		if( ( fabs(part_z) - fabs(part_z_previous) ) < (0 + tolerance_space) ) {
		  Output << "\t ERROR |z| ordering (front/back) not respected in ring ";
		  iERROR++;
		} else {
		  Output << "\t OK" << " |z| ordering (front/back) in ring ";
		  iOK++;
		}
		Output << iRing << " part " << iPart+1 << " to " << iPart << " (" << part_z_previous << " --> " << part_z << ")" << std::endl;
		//
		part_z_previous = part_z;
		module_phi_previous = -M_PI;
		//
	      } // part loop
	      
	      // ring: R check
	      Output << "\t ### ";
	      if( ( ring_R - ring_R_previous ) < (0 + tolerance_space) ) {
		Output << "\t ERROR R ordering not respected in disk ";
		iERROR++;
	      } else {
		Output << "\t OK" << " R ordering in disk ";
		iOK++;
	      }
	      Output << iDisk << " ring " << iRing-1 << " to " << iRing << " (" << ring_R_previous << " --> " << ring_R << ")" << std::endl;
	      //
	      ring_R_previous     = ring_R;
	      part_z_previous     = 0.0;
	      module_phi_previous = -M_PI;
	      //
	    } // ring loop
	    
	    // disk: |z| check
	    Output << "\t #### ";
	    if( ( fabs(disk_z) - fabs(disk_z_previous) ) < (0 + tolerance_space) ) {
	      Output << "\t ERROR |z| ordering not respected in disk ";
	      iERROR++;
	    } else {
	      Output << "\t OK" << " |z| ordering in disk ";
	      iOK++;
	    }
	    Output << iDisk-1 << " to " << iDisk << " (" << disk_z_previous << " --> " << disk_z << ")" << std::endl;	    
	    //
	    disk_z_previous     = disk_z;
	    ring_R_previous     = 0.0;
	    part_z_previous     = 0.0;
	    module_phi_previous = -M_PI;
	    //
	  } // disk loop
	  
	  // side: z check
	  Output << "\t ##### ";
	  if( ( side_z - side_z_previous ) > (0 + tolerance_space) ) {
	    Output << "\t ERROR z ordering (side -/+) not respected in TID side ";
	    iERROR++;
	  } else {
	    Output << "\t OK" << " z ordering (side -/+) in TID side ";
	    iOK++;
	  }
	  Output << iSide+1 << " to " << iSide << " (" << side_z_previous << " --> " << side_z << ")" << std::endl;
	  //
	  side_z_previous     = side_z;
	  disk_z_previous     = 0.0;
	  ring_R_previous     = 0.0;
	  part_z_previous     = 0.0;
	  module_phi_previous = -M_PI;
	  //	    
	} // side loop
	
	break;
      }
      
      // TOB
    case 5:
      {
	// TOB loop
	unsigned int rod_TOB[8] = { 42 , 48 , 54 , 60 , 66, 74 }; // number of rods per layer 1/6
	unsigned int mod_type_TOB[12] = { 1 , 2 , 1 , 2 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 }; // first and last type for module type loop
	unsigned int nLayers  = 6;
	unsigned int nModules = 6;
	// debug variables
	double layer_R           = 0.0;
	double layer_R_previous  = 0.0;
	double side_z            = 0.0;
	double side_z_previous   = -10000.0;
	double rod_phi           = 0.0;
	double rod_phi_previous  = 0.0;
	double module_z          = 0.0;
	double module_z_previous = 0.0;
	//
	for(unsigned int iLayer = 1; iLayer <= nLayers; iLayer++) { // Layer: 1,...,nLayers
	  for(unsigned int iSide = 1; iSide <= 2; iSide++){ // Side: 1:- 2:+
	    for(unsigned int iRod = 1; iRod <= rod_TOB[iLayer-1] ; iRod++) { // Rod: 1,...,nRods
	      for(unsigned int iModule = 1; iModule <= nModules ; iModule++) { // Module: 1,...,nModules
		for(unsigned int iType = mod_type_TOB[2*(iLayer-1)]; iType <= mod_type_TOB[2*(iLayer-1)+1] ; iType++) { // Module Type: 0 (ss) 1-2 (ds stereo and rphi)
		  
		  myDetId = 0;
		  // detector
		  myDetId <<= 4;
		  myDetId |= iDetector;
		  // subdetector
		  myDetId <<= 3;
		  myDetId |= iSubDetector;
		  // not used
		  myDetId <<= 8;
		  myDetId |= 0;
		  // layer
		  myDetId <<= 3;
		  myDetId |= iLayer;
		  // side
		  myDetId <<= 2;
		  myDetId |= iSide;
		  // rod number
		  myDetId <<= 7;
		  myDetId |= iRod;
		  // module number
		  myDetId <<= 3;
		  myDetId |= iModule;
		  // module type
		  myDetId <<= 2;
		  myDetId |= iType;
		  //
		  std::bitset<32> binary_myDetId(myDetId);
		  Output << std::endl  << std::endl;
		  Output << " ******** myDet Id = " << myDetId << " (" << binary_myDetId << ")" << std::endl;
		  //
		  unsigned int rawid = mapDetIdToGeometricDet[myDetId]->geographicalID().rawId();
		  std::bitset<32> binary_detid(rawid);
		  GeometricDet::nav_type detNavType = mapDetIdToGeometricDet[myDetId]->navType();
		  //
		  Output << "            raw Id = " << rawid << " (" << binary_detid << ")"
			 << "\t nav type = " << printNavType(Output,&detNavType.front(),detNavType.size()) << std::endl;
		  
		  // variables
		  fillModuleVariables(mapDetIdToGeometricDet[myDetId], polarRadius, phiRad, z);
		  layer_R  = polarRadius;
		  side_z   = z;
		  rod_phi  = phiRad;
		  module_z = z;
		  //
		  
		  Output << "\t R = " << polarRadius << "\t phi = " << phiRad << "\t z = " << z << std::endl;
		  
		  // Module Info
		  
		  std::string name = mapDetIdToGeometricDet[myDetId]->name().name();
		  unsigned int         theLayer  = tTopo->tobLayer(rawid);
		  std::vector<unsigned int> theRod    = tTopo->tobRodInfo(rawid);
		  unsigned int         theModule = tTopo->tobModule(rawid);
		  std::string side;
		  std::string part;
		  side = (theRod[0] == 1 ) ? "-" : "+";
		  Output << " TOB" << side << "\t" << "Layer " << theLayer 
			 << "\t" << "rod " << theRod[1] << " module " << theModule << "\t" << name << std::endl;
		  //
		  
		  // module: |z| check
		  Output << "\t # ";
		  if( ( fabs(module_z) - fabs(module_z_previous) ) < (0 + tolerance_space) ) {
		    Output << "\t ERROR |z| ordering not respected in module ";
		    iERROR++;
		  } else {
		    Output << "\t OK" << " |z| ordering in module ";
		    iOK++;
		  }
		  Output << iModule-1 << " to " << iModule << " (" << module_z_previous << " --> " << module_z << ")" << std::endl;
		  //
		} // type loop
		
		//
		module_z_previous = module_z;
		//
	      } // module loop
	      
	      // rod: phi check
	      Output << "\t ## ";
	      if( ( rod_phi - rod_phi_previous ) < (0 - tolerance_angle) ) {
		Output << "\t ERROR phi ordering not respected in rod ";
		iERROR++;
	      } else {
		Output << "\t OK" << " phi ordering in rod ";
		iOK++;
	      }
	      Output << iRod-1 << " to " << iRod << " (" << rod_phi_previous << " --> " << rod_phi << ")" << std::endl;
	      //
	      rod_phi_previous  = rod_phi;
	      module_z_previous = 0.0;
	      //		
	      } // rod loop
	    
	    // side: z check
	    Output << "\t ### ";
	    if( ( side_z - side_z_previous ) < (0 + tolerance_space) ) {
	      Output << "\t ERROR z ordering (side -/+) not respected in layer ";
	      iERROR++;
	    } else {
	      Output << "\t OK" << " z ordering (side -/+) in layer ";
	      iOK++;
	    }
	    Output << iLayer << " side " << iSide-1 << " to " << iSide << " (" << side_z_previous << " --> " << side_z << ")" << std::endl;	      
	    //
	    side_z_previous   = side_z;
	    rod_phi_previous  = 0.0;
	    module_z_previous = 0.0;
	    //	    
	  } // side loop
	  
	  // layer: R check
	  Output << "\t #### ";
	  if( ( layer_R - layer_R_previous ) < (0 + tolerance_space) ) {
	    Output << "\t ERROR R ordering not respected from layer ";
	    iERROR++;
	  } else {
	    Output << "\t OK" << " R ordering in layer ";
	    iOK++;
	  }
	  Output << iLayer-1 << " to " << iLayer << " (" << layer_R_previous << " --> " << layer_R << ")" << std::endl;	    
	  //
	  layer_R_previous  = layer_R;
	  side_z_previous   = -10000.0;
	  rod_phi_previous  = 0.0;
	  module_z_previous = 0.0;
	//
	} // layer loop

	break;
      }
      
      // TEC
    case 6:
      {

	// TEC loop
	unsigned int nWheels = 9;
	unsigned int nPetals = 8;
	unsigned int nRings  = 7;
	unsigned int first_ring_TEC[9] = { 1 , 1 , 1 , 2 , 2 , 2 , 3 , 3 , 4 }; // first ring of the wheel
	unsigned int modules_ring_TEC[14] = { 1 , 2 , 1 , 2 , 2 , 3 , 3 , 4 , 3 , 2 , 3 , 4 , 5 , 5 }; // number of modules ring=1,...,nRings back/front
	unsigned int mod_type_TEC[14] = { 1 , 2 , 1 , 2 , 0 , 0 , 0 , 0 , 1 , 2 , 0 , 0 , 0 , 0 }; // first and last type for module type loop (per ring)
	// debug variables
	double side_z              = 0.0;
	double side_z_previous     = 10000.0;
	double wheel_z             = 0.0;  
	double wheel_z_previous    = 0.0;  
	double part_z              = 0.0;  
	double part_z_previous     = 0.0;  
	double petal_phi           = 0.0;
	double petal_phi_previous  = 0.0;
	double ring_R              = 0.0;
	double ring_R_previous     = 0.0;
	double module_phi          = 0.0;
	double module_phi_previous = -M_PI;
	//
	for(unsigned int iSide = 2; iSide >= 1; iSide--){ // Side: 1:- 2:+
	  switch (iSide) {
	    // TEC+
	  case 2:
	    {
	      side_z              = 0.0;
	      side_z_previous     = 10000.0;
	      wheel_z             = 0.0;  
	      wheel_z_previous    = 0.0;  
	      part_z              = 0.0;  
	      part_z_previous     = 0.0;  
	      petal_phi           = 0.0;
	      petal_phi_previous  = 0.0;
	      ring_R              = 0.0;
	      ring_R_previous     = 0.0;
	      module_phi          = 0.0;
	      module_phi_previous = -M_PI;
	      break;
	    }
	    // TEC-
	  case 1:
	    {
	      wheel_z             = 0.0;  
	      wheel_z_previous    = 0.0;  
	      part_z              = 0.0;  
	      part_z_previous     = 0.0;  
	      petal_phi           = 0.0;
	      petal_phi_previous  = 0.0;
	      ring_R              = 0.0;
	      ring_R_previous     = 0.0;
	      module_phi          = 0.0;
	      module_phi_previous = 2*M_PI;
	      break;
	    }
	  default:
	    {
	      // do nothing
	    }
	  }
	  //
	  for(unsigned int iWheel = 1; iWheel <= nWheels; iWheel++) { // Wheel: 1,...,nWheels
	    for(int iPart = 2; iPart >= 1; iPart--){ // Part: 1:back 2:front
	      for(unsigned int iPetal = 1; iPetal <= nPetals; iPetal++) { // Petal: 1,...,nPetals
		for(unsigned int iRing = first_ring_TEC[iWheel-1]; iRing <= nRings; iRing++){ // Ring: first,...,nRings
		  unsigned int nModules = modules_ring_TEC[2*(iRing-1)+(iPart-1)];
		  for(unsigned int iModule = 1; iModule <= nModules; iModule++) {  // Module: 1,...,modules in ring of petal
		    for(unsigned int iType = mod_type_TEC[2*(iRing-1)]; iType <= mod_type_TEC[2*(iRing-1)+1] ; iType++) { // Module Type: 0 (ss) 1-2 (ds stereo and rphi)
		      
		      myDetId = 0;
		      // detector
		      myDetId <<= 4;
		      myDetId |= iDetector;
		      // subdetector
		      myDetId <<= 3;
		      myDetId |= iSubDetector;
		      // not used
		      myDetId <<= 5;
		      myDetId |= 0;
		      // side
		      myDetId <<= 2;
		      myDetId |= iSide;
		      // wheel number
		      myDetId <<= 4;
		      myDetId |= iWheel;
		      // part
		      myDetId <<= 2;
		      myDetId |= iPart;
		      // petal number
		      myDetId <<= 4;
		      myDetId |= iPetal;
		      // ring number
		      myDetId <<= 3;
		      myDetId |= iRing;
		      // module number
		      myDetId <<= 3;
		      myDetId |= iModule;
		      // module type
		      myDetId <<= 2;
		      myDetId |= iType;
		      //
		      std::bitset<32> binary_myDetId(myDetId);
		      Output << std::endl  << std::endl;
		      Output << " ******** myDet Id = " << myDetId << " (" << binary_myDetId << ")" << std::endl;
		      //
		      unsigned int rawid = mapDetIdToGeometricDet[myDetId]->geographicalID().rawId();
		      std::bitset<32> binary_detid(rawid);
		      GeometricDet::nav_type detNavType = mapDetIdToGeometricDet[myDetId]->navType();
		      //
		      Output << "            raw Id = " << rawid << " (" << binary_detid << ")"
			     << "\t nav type = " << printNavType(Output,&detNavType.front(),detNavType.size()) << std::endl;
		      
		      // variables
		      fillModuleVariables(mapDetIdToGeometricDet[myDetId], polarRadius, phiRad, z);
		      side_z     = z;
		      wheel_z    = z;  
		      part_z     = z;
		      switch (iSide) {
			// TEC+
		      case 2:
			{
			  phiRad = phiRad;
			  break;
			}
			// TEC-
		      case 1:
			{
			  phiRad = changePhiRange_From_ZeroTwoPi_To_MinusPiPlusPi(phiRad);
			  break;
			}
		      default:
			{
			  // do nothing
			}
		      }
		      
		      // petal must be ordered inside [0,2PI) for TEC+, [PI,-PI) for TEC-, take the phi of the central module in a ring with (2n+1) modules
		      if( ( nModules%2 ) && ( iModule==(int)(nModules/2)+(nModules%2) ) ) {
			switch (iSide) {
			  // TEC+
			case 2:
			  {
			    petal_phi = phiRad;
			    break;
			  }
			  // TEC-
			case 1:
			  {
			    petal_phi = changePhiRange_From_MinusPiPlusPi_To_ZeroTwoPi(phiRad);
			    break;
			  }
			default:
			  {
			    // do nothing
			  }
			}
		      }
		      
		      ring_R     = polarRadius;
		      //		      
		      // modules must be ordered inside petals [0,2PI)-->[-PI,PI) if the petal is near phi~0  TEC+ (petal number 1)
		      // modules must be ordered inside petals [PI,-PI)-->[2PI,0) if the petal is near phi~PI TEC- (petal number 5)
		      switch (iSide) {
			// TEC+
		      case 2:
			{
			  if( iPetal == 1 ) { // it is the request of the petal at phi = 0, always the first
			    module_phi = changePhiRange_From_ZeroTwoPi_To_MinusPiPlusPi(phiRad);
			  } else {
			    module_phi = phiRad;
			  }
			  break;
			}
			// TEC-
		      case 1:
			{ 
			  if( iPetal == 5 ) { // it is the request of the petal at phi = PI, always the fifth
			    module_phi = changePhiRange_From_MinusPiPlusPi_To_MinusTwoPiZero(phiRad);
			  } else {
			    module_phi = phiRad;
			  }
			  break;
			}
		      default:
			{
			  // do nothing
			}
		      }
		      
		      //
		      
		      Output << "\t R = " << polarRadius << "\t phi = " << phiRad << "\t z = " << z << std::endl;
		      
		      // Module Info
		      
		      std::string name = mapDetIdToGeometricDet[myDetId]->name().name();
		      unsigned int theWheel = tTopo->tecWheel(rawid);
		      unsigned int         theModule = tTopo->tecModule(rawid);
		      std::vector<unsigned int> thePetal  = tTopo->tecPetalInfo(rawid);
		      unsigned int         theRing   = tTopo->tecRing(rawid);
		      std::string side;
		      std::string petal;
		      side  = (tTopo->tecSide(rawid) == 1 ) ? "-" : "+";
		      petal = (thePetal[0] == 1 ) ? "back" : "front";
		      Output << " TEC" << side << "\t" << "Wheel " << theWheel << " Petal " << thePetal[1] << " " << petal << " Ring " << theRing << "\t"
			     << "\t" << " module " << theModule << "\t" << name << std::endl;
		      //
		      
		      // module: phi check
		      Output << "\t # ";
		      switch (iSide) {
			// TEC+
		      case 2:
			{
			  if( ( module_phi - module_phi_previous ) < (0 - tolerance_angle) ) {
			    Output << "\t ERROR phi ordering not respected in module ";
			    iERROR++;
			  } else {
			    Output << "\t OK" << " phi ordering in module ";
			    iOK++;
			  }
			  Output << iModule-1 << " to " << iModule << " (" << module_phi_previous << " --> " << module_phi << ")" << std::endl;
			  break;
			}
			// TEC-
		      case 1:
			{
			  if( ( module_phi - module_phi_previous ) > (0 + tolerance_angle) ) {
			    Output << "\t ERROR phi ordering not respected in module ";
			    iERROR++;
			  } else {
			    Output << "\t OK" << " phi ordering in module ";
			    iOK++;
			  }
			  Output << iModule-1 << " to " << iModule << " (" << module_phi_previous << " --> " << module_phi << ")" << std::endl;
			  break;
			}
		      default:
			{
			  // do nothing
			}
		      }
		      //
		    } // type loop
		    
		    //
		    module_phi_previous = module_phi;
		    //
		  } // module loop
		  
		  // ring: R check
		  Output << "\t ## ";
		  if( ( ring_R - ring_R_previous ) < (0 + tolerance_space) ) {
		    Output << "\t ERROR R ordering not respected in petal ";
		    iERROR++;
		  } else {
		    Output << "\t OK" << " R ordering in petal ";
		    iOK++;
		  }
		  Output << iPetal << " ring " << iRing-1 << " to " << iRing << " (" << ring_R_previous << " --> " << ring_R << ")" << std::endl;
		  //
		  switch (iSide) {
		    // TEC+
		  case 2:
		    {
		      ring_R_previous     = ring_R;
		      module_phi_previous = -M_PI;
		      break;
		    }
		    // TEC-
		  case 1:
		    {
		      ring_R_previous     = ring_R;
		      module_phi_previous = 2*M_PI;
		      break;
		    }
		  default:
		    {
		      // do nothing
		    }
		  }
		  //
		} // ring loop
		
		// petal: phi check
		Output << "\t ### ";
		switch (iSide) {
		  // TEC+
		case 2:
		  {
		    if( ( petal_phi - petal_phi_previous ) < (0 - tolerance_angle) ) {
		      Output << "\t ERROR phi ordering not respected in petal ";
		      iERROR++;
		    } else {
		      Output << "\t OK" << " phi ordering in petal ";
		      iOK++;
		    }
		    Output << iPetal-1 << " to " << iPetal << " (" << petal_phi_previous << " --> " << petal_phi << ")" << std::endl;
		    //
		    petal_phi_previous  = petal_phi;
		    ring_R_previous     = 0.0;
		    module_phi_previous = -M_PI;
		    //
		    break;
		  }
		  // TEC-
		case 1:
		  {
		    if( ( petal_phi - petal_phi_previous ) < (0 - tolerance_angle) ) {
		      Output << "\t ERROR phi ordering not respected in petal ";
		      iERROR++;
		    } else {
		      Output << "\t OK" << " phi ordering in petal ";
		      iOK++;
		    }
		    Output << iPetal-1 << " to " << iPetal << " (" << petal_phi_previous << " --> " << petal_phi << ")" << std::endl;
		    //
		    petal_phi_previous  = petal_phi;
		    ring_R_previous     = 0.0;
		    module_phi_previous = 2*M_PI;
		    //
		    break;
		  }
		default:
		  {
		    // do nothing
		  }
		}
		//
	      } // petal loop
	      
	      // part: |z| check
	      Output << "\t #### ";
	      if( ( fabs(part_z) - fabs(part_z_previous) ) < (0 + tolerance_space) ) {
		Output << "\t ERROR |z| ordering (front/back) not respected in wheel ";
		iERROR++;
	      } else {
		Output << "\t OK" << " |z| (front/back) ordering in wheel ";
		iOK++;
	      }
	      Output << iWheel << " part " << iPart+1 << " to " << iPart << " (" << part_z_previous << " --> " << part_z << ")" << std::endl;
	      //
	      switch (iSide) {
		// TEC+
	      case 2:
		{
		  part_z_previous     = part_z;
		  petal_phi_previous  = 0.0;
		  ring_R_previous     = 0.0;
		  module_phi_previous = -M_PI;
		  break;
		}
		// TEC-
	      case 1:
		{
		  part_z_previous     = part_z;
		  petal_phi_previous  = 0.0;
		  ring_R_previous     = 0.0;
		  module_phi_previous = 2*M_PI;
		  break;
		}
	      default:
		{
		  // do nothing
		}
	      }
	      //
	    } // part loop
	    
	    // wheel: |z| check
	    Output << "\t ##### ";
	    if( ( fabs(wheel_z) - fabs(wheel_z_previous) ) < (0 + tolerance_space) ) {
	      Output << "\t ERROR |z| ordering not respected in wheel ";
	      iERROR++;
	    } else {
	      Output << "\t OK" << " |z| ordering in wheel ";
	      iOK++;
	    }
	    Output << iWheel-1 << " to " << iWheel << " (" << wheel_z_previous << " --> " << wheel_z << ")" << std::endl;	    
	    //
	    switch (iSide) {
	      // TEC+
	    case 2:
	      {
		wheel_z_previous    = wheel_z;
		part_z_previous     = 0.0;  
		petal_phi_previous  = 0.0;
		ring_R_previous     = 0.0;
		module_phi_previous = -M_PI;
		break;
	      }
	      // TEC-
	    case 1:
	      {
		wheel_z_previous    = wheel_z;
		part_z_previous     = 0.0;
		petal_phi_previous  = 0.0;
		ring_R_previous     = 0.0;
		module_phi_previous = 2*M_PI;
		break;
	      }
	    default:
	      {
		// do nothing
	      }
	    }
	    //
	  } // wheel loop
	  
	  // side: z check
	  Output << "\t ###### ";
	  if( ( side_z - side_z_previous ) > (0 + tolerance_space) ) {
	    Output << "\t ERROR z ordering (side -/+) not respected in TEC side ";
	    iERROR++;
	  } else {
	    Output << "\t OK" << " z ordering (side -/+) in TEC side ";
	    iOK++;
	  }
	  Output << iSide+1 << " to " << iSide << " (" << side_z_previous << " --> " << side_z << ")" << std::endl;
	  //
	  switch (iSide) {
	    // TEC+
	  case 2:
	    {
	      side_z_previous     = side_z;
	      wheel_z_previous    = 0.0;
	      part_z_previous     = 0.0;  
	      petal_phi_previous  = 0.0;
	      ring_R_previous     = 0.0;
	      module_phi_previous = -M_PI;
	      break;
	    }
	    // TEC-
	  case 1:
            {
	      side_z_previous     = side_z;
	      wheel_z_previous    = 0.0;
	      part_z_previous     = 0.0;  
	      petal_phi_previous  = 0.0;
	      ring_R_previous     = 0.0;
	      module_phi_previous = 2*M_PI;
	      break;
	    }
	  default:
	    {
	      // do nothing
	    }
	  }
	  //	    
	} // side loop
	
	break;
      }
    default:
      Output << " WARNING no Silicon Strip subdetector, I got a " << iSubDetector << std::endl;;
    }
    
    
  } // subdetector loop
  
  // summary
  unsigned int iChecks = iOK + iERROR;
  Output << std::endl << std::endl;
  Output << "-------------------------------------" << std::endl;
  Output << " Module Numbering Check Summary      " << std::endl;
  Output << "-------------------------------------" << std::endl;
  Output << " Number of checks:   " << std::setw(6) << iChecks << std::endl;
  Output << "               OK:   " << std::setw(6) << iOK
	 << " (" << std::fixed << std::setprecision(2) << ((double)iOK    / (double)iChecks) * 100 << "%) " << std::endl;
  Output << "           ERRORS:   " << std::setw(6) << iERROR
	 << " (" << std::fixed << std::setprecision(2) << ((double)iERROR / (double)iChecks) * 100 << "%) " << std::endl;
  Output << "-------------------------------------" << std::endl;
  
}


//define this as a plug-in
DEFINE_FWK_MODULE(ModuleNumbering);
  
