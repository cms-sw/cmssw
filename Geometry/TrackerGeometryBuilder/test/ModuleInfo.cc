// -*- C++ -*-
//
/* 
 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

//
// Original Author:  Riccardo Ranieri
//         Created:  Wed May 3 10:30:00 CEST 2006
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
#include "Geometry/Surface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
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

//
//
// class decleration
//

class ModuleInfo : public edm::EDAnalyzer {
public:
  explicit ModuleInfo( const edm::ParameterSet& );
  ~ModuleInfo();
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  static const double density_units = 6.24151e+18;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ModuleInfo::ModuleInfo( const edm::ParameterSet& iConfig )
{
  //now do what ever initialization is needed
  
}


ModuleInfo::~ModuleInfo()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ModuleInfo::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;
  using namespace std;
  
  edm::LogInfo("ModuleInfo") << "begins";
  
  // output file
  std::ofstream Output("ModuleInfo.log",ios::out);
  // TEC output as Martin Weber's
  std::ofstream TECOutput("TECLayout_CMSSW.dat",ios::out);
  //
  
  //
  // get the GeometricDet
  //
  edm::ESHandle<GeometricDet> rDD;
  iSetup.get<IdealGeometryRecord>().get( rDD );     
  edm::LogInfo("ModuleInfo") << " Top node is  " << &(*rDD) << " " <<  (*rDD).name().name() << endl;
  edm::LogInfo("ModuleInfo") << " And Contains  Daughters: " << (*rDD).deepComponents().size() << endl;
  CmsTrackerDebugNavigator nav;
  nav.dump(&(*rDD));
  //
  //first instance tracking geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
  //
  
  // counters
  unsigned int tibN = 0;
  unsigned int tib_L12_rphiN = 0;
  unsigned int tib_L12_sterN = 0;
  unsigned int tib_L34_rphiN = 0;
  unsigned int tidN = 0;
  unsigned int tid_r1_rphiN = 0;
  unsigned int tid_r1_sterN = 0;
  unsigned int tid_r2_rphiN = 0;
  unsigned int tid_r2_sterN = 0;
  unsigned int tid_r3_rphiN = 0;
  unsigned int tobN = 0;
  unsigned int tob_R12_rphiN = 0;
  unsigned int tob_R12_sterN = 0;
  unsigned int tob_R34_rphiN = 0;
  unsigned int tob_R56_rphiN = 0;
  unsigned int tecN = 0;
  unsigned int tec_r1_rphiN = 0;
  unsigned int tec_r1_sterN = 0;
  unsigned int tec_r2_rphiN = 0;
  unsigned int tec_r2_sterN = 0;
  unsigned int tec_r3_rphiN = 0;
  unsigned int tec_r4_rphiN = 0;
  unsigned int tec_r5_rphiN = 0;
  unsigned int tec_r5_sterN = 0;
  unsigned int tec_r6_rphiN = 0;
  unsigned int tec_r7_rphiN = 0;
  //
  double volume_total        = 0.0;
  double weight_total        = 0.0;
  double activeSurface_total = 0.0;
  double volume_tib          = 0.0;
  double weight_tib          = 0.0;
  double activeSurface_tib   = 0.0;
  double volume_tid          = 0.0;
  double weight_tid          = 0.0;
  double activeSurface_tid   = 0.0;
  double volume_tob          = 0.0;
  double weight_tob          = 0.0;
  double activeSurface_tob   = 0.0;
  double volume_tec          = 0.0;
  double weight_tec          = 0.0;
  double activeSurface_tec   = 0.0;
  //
  
  
  vector<const GeometricDet*> modules =  (*rDD).deepComponents();
  Output << "************************ List of modules with positions ************************" << endl;
  for(unsigned int i=0; i<modules.size();i++){
    unsigned int rawid = modules[i]->geographicalID().rawId();
    Output << " ******** raw Id = " << rawid << endl;
    int subdetid = modules[i]->geographicalID().subdetId();
    double volume = modules[i]->volume() / 1000; // mm3->cm3
    double density = modules[i]->density() / density_units;
    double weight = modules[i]->weight() / density_units / 1000.; // [kg], hence the factor 1000;
    double thickness = modules[i]->bounds()->thickness() * 10000; // cm-->um
    double activeSurface = volume / ( thickness / 10000 ); // cm2 (thickness in um)
    volume_total+=volume;
    weight_total+=weight;
    activeSurface_total+=activeSurface;
    switch (subdetid) {
    case 3:
      {
	tibN++;
	volume_tib+=volume;
	weight_tib+=weight;
	activeSurface_tib+=activeSurface;
	string name = modules[i]->name().name();
	if(name == "TIBActiveRphi0") tib_L12_rphiN++;
	if(name == "TIBActiveSter0") tib_L12_sterN++;
	if(name == "TIBActiveRphi2") tib_L34_rphiN++;
	TIBDetId module(rawid);
	unsigned int         theLayer  = module.layer();
	vector<unsigned int> theString = module.string();
	unsigned int         theModule = module.module();
	string side;
	string part;
	side = (theString[0] == 0 ) ? "-" : "+";
	part = (theString[1] == 0 ) ? "int" : "ext";
	
	Output << " TIB" << side << "\t" << "Layer " << theLayer << " " << part
	       << "\t" << "string " << theString[2] << "\t" << " module " << theModule << " " << name << "\t"
	       << "son of " << modules[i]->parents()[modules[i]->parents().size()-3].logicalPart().name() << " "
	       << modules[i]->translation()[0] << "   \t" << modules[i]->translation()[1] << "   \t" << modules[i]->translation()[2]
		 << "\t" << "polar radius " 
		 << sqrt(modules[i]->translation()[0]*modules[i]->translation()[0]+modules[i]->translation()[1]*modules[i]->translation()[1])
		 << endl
		 << "\t" << "volume " << volume << " cm3 \t" << "density " << density << " g/cm3 \t" << "weight " << weight << " kg \t" << "thickness " << thickness
		 << " um \t" << " active area " << activeSurface << " cm2" << endl;

	break;
      }
    case 4:
      {
	tidN++;      
	volume_tid+=volume;
	weight_tid+=weight;
	activeSurface_tid+=activeSurface;
	string name = modules[i]->name().name();
	if(name == "TIDModule0RphiActive")   tid_r1_rphiN++;
	if(name == "TIDModule0StereoActive") tid_r1_sterN++;
	if(name == "TIDModule1RphiActive")   tid_r2_rphiN++;
	if(name == "TIDModule1StereoActive") tid_r2_sterN++;
	if(name == "TIDModule2RphiActive")   tid_r3_rphiN++;
	TIDDetId module(rawid);
	unsigned int         theDisk   = module.wheel();
	unsigned int         theRing   = module.ring();
	vector<unsigned int> theModule = module.module();
	string side;
	string part;
	side = (module.side() == 1 ) ? "-" : "+";
	part = (theModule[0] == 0 ) ? "back" : "front";
	Output << " TID" << side << "\t" << "Disk " << theDisk << " Ring " << theRing << " " << part
	       << "\t" << " module " << theModule[1] << "\t" << name << "\t"
	       << "son of " << modules[i]->parents()[modules[i]->parents().size()-3].logicalPart().name() << " "
	       << modules[i]->translation()[0] << "   \t" << modules[i]->translation()[1] << "   \t" << modules[i]->translation()[2]
	  //	     <<	"   \t" << theModule[0] << theModule[1]
	       << "\t" << "polar radius " 
	       << sqrt(modules[i]->translation()[0]*modules[i]->translation()[0]+modules[i]->translation()[1]*modules[i]->translation()[1])
	       << endl
	       << "\t" << "volume " << volume << " cm3 \t" << "density " << density << " g/cm3 \t" << "weight " << weight << " kg \t" << "thickness " << thickness
	       << " um \t" << " active area " << activeSurface << " cm2" << endl;
	break;
      }
    case 5:
      {
	tobN++;
	volume_tob+=volume;
	weight_tob+=weight;
	activeSurface_tob+=activeSurface;
	string name = modules[i]->name().name();
	if(name == "TOBActiveRphi0") tob_R12_rphiN++;
	if(name == "TOBActiveSter0") tob_R12_sterN++;
	if(name == "TOBActiveRphi2") tob_R34_rphiN++;
	if(name == "TOBActiveRphi4") tob_R56_rphiN++;
	TOBDetId module(rawid);
	unsigned int         theLayer  = module.layer();
	vector<unsigned int> theRod    = module.rod();
	unsigned int         theModule = module.module();
	string side;
	string part;
	side = (theRod[0] == 1 ) ? "+" : "-";
	
	Output << " TOB" << side << "\t" << "Layer " << theLayer 
	       << "\t" << "rod " << theRod[1] << " module " << theModule << "\t" << name << "\t" 
	       << "son of " << modules[i]->parents()[modules[i]->parents().size()-3].logicalPart().name() << " "
	       << modules[i]->translation()[0] << "   \t" << modules[i]->translation()[1] << "   \t" << modules[i]->translation()[2]
	       << "\t" << "polar radius " 
	       << sqrt(modules[i]->translation()[0]*modules[i]->translation()[0]+modules[i]->translation()[1]*modules[i]->translation()[1])
	       << "\t" << "phi [deg] " 
	       << atan2(modules[i]->translation()[1],modules[i]->translation()[0]) * 360. / 6.283185307
	       << " rod name " << modules[i]->parents()[modules[i]->parents().size()-6].logicalPart().name()
	       << " copy number " << modules[i]->parents()[modules[i]->parents().size()-6].copyno()
	       << endl
	       << "\t" << "volume " << volume << " cm3 \t" << "density " << density << " g/cm3 \t" << "weight " << weight << " kg \t" << "thickness " << thickness
	       << " um \t" << " active area " << activeSurface << " cm2" << endl;
	break;
      }
    case 6:
      {
	tecN++;      
	volume_tec+=volume;
	weight_tec+=weight;
	activeSurface_tec+=activeSurface;
	string name = modules[i]->name().name();
	if(name == "TECModule0RphiActive")   tec_r1_rphiN++;
	if(name == "TECModule0StereoActive") tec_r1_sterN++;
	if(name == "TECModule1RphiActive")   tec_r2_rphiN++;
	if(name == "TECModule1StereoActive") tec_r2_sterN++;
	if(name == "TECModule2RphiActive")   tec_r3_rphiN++;
	if(name == "TECModule3RphiActive")   tec_r4_rphiN++;
	if(name == "TECModule4RphiActive")   tec_r5_rphiN++;
	if(name == "TECModule4StereoActive") tec_r5_sterN++;
	if(name == "TECModule5RphiActive")   tec_r6_rphiN++;
	if(name == "TECModule6RphiActive")   tec_r7_rphiN++;
	TECDetId module(rawid);
	unsigned int theWheel = module.wheel();
	unsigned int         theModule = module.module();
	vector<unsigned int> thePetal  = module.petal();
	unsigned int         theRing   = module.ring();
	string side;
	string petal;
	side  = (module.side() == 1 ) ? "-" : "+";
	petal = (thePetal[0] == 1 ) ? "forward" : "backward";
	Output << " TEC" << side << "\t" << "Wheel " << theWheel << " Petal " << thePetal[1] << " " << petal << " Ring " << theRing << "\t"
	       << "\t" << " module " << theModule << "\t" << name << "\t"
	       << "son of " << modules[i]->parents()[modules[i]->parents().size()-3].logicalPart().name() << " "
	       << modules[i]->translation()[0] << "   \t" << modules[i]->translation()[1] << "   \t" << modules[i]->translation()[2]
	       << "\t" << "polar radius " 
	       << sqrt(modules[i]->translation()[0]*modules[i]->translation()[0]+modules[i]->translation()[1]*modules[i]->translation()[1])
	       << "\t" << "phi [deg] " 
	       << atan2(modules[i]->translation()[1],modules[i]->translation()[0]) * 360. / 6.283185307
	       << "\t" << "phi [rad] "
	       << atan2(modules[i]->translation()[1],modules[i]->translation()[0])
	       << endl
	       << "\t" << "volume " << volume << " cm3 \t" << "density " << density << " g/cm3 \t" << "weight " << weight << " kg \t" << "thickness " << thickness
	       << " um \t" << " active area " << activeSurface << " cm2" << endl;
	
	// TEC output as Martin Weber's
	int out_side  = (module.side() == 1 ) ? -1 : 1;
	unsigned int out_disk = module.wheel();
	unsigned int out_sector = thePetal[1];
	int out_petal = (thePetal[0] == 1 ) ? -1 : 1;
	unsigned int out_ring = module.ring();
	unsigned int out_module = module.module();
	int out_sensor = 0;
	if(name == "TECModule0RphiActive")   out_sensor = -1;
	if(name == "TECModule0StereoActive") out_sensor =  1;
	if(name == "TECModule1RphiActive")   out_sensor = -1;
	if(name == "TECModule1StereoActive") out_sensor =  1;
	if(name == "TECModule2RphiActive")   out_sensor = -1;
	if(name == "TECModule3RphiActive")   out_sensor = -1;
	if(name == "TECModule4RphiActive")   out_sensor = -1;
	if(name == "TECModule4StereoActive") out_sensor =  1;
	if(name == "TECModule5RphiActive")   out_sensor = -1;
	if(name == "TECModule6RphiActive")   out_sensor = -1;
	double out_x = modules[i]->translation()[0];
	double out_y = modules[i]->translation()[1];
	double out_z = modules[i]->translation()[2];
	double out_r = sqrt(modules[i]->translation()[0]*modules[i]->translation()[0] + 
			    modules[i]->translation()[1]*modules[i]->translation()[1]);
	double out_phi_rad = atan2(modules[i]->translation()[1],modules[i]->translation()[0]);
	TECOutput << out_side << " " << out_disk << " " << out_sector << " " << out_petal
		  << " " << out_ring << " " << out_module << " " << out_sensor
		  << " " << out_x << " " << out_y << " " << out_z << " " << out_r << " " << out_phi_rad << endl;
	//
	break;
      }
    default:
      Output << " WARNING no Silicon Strip detector, I got a " << rawid << endl;;
    }
    
    // Local axes from Reco
    const GeomDet* geomdet = pDD->idToDet(modules[i]->geographicalID());
    // Global Coordinates (i,j,k)
    LocalVector xLocal(1,0,0);
    LocalVector yLocal(0,1,0);
    LocalVector zLocal(0,0,1);
    // Versor components
    GlobalVector xGlobal = (geomdet->surface()).toGlobal(xLocal);
    GlobalVector yGlobal = (geomdet->surface()).toGlobal(yLocal);
    GlobalVector zGlobal = (geomdet->surface()).toGlobal(zLocal);
    //
    
    // sensor versors
    Output << "\t z = n = (" << modules[i]->rotation().xz() << "," << modules[i]->rotation().yz() << "," << modules[i]->rotation().zz() << ")" << endl
	   << "\t [Rec] = (" << zGlobal.x()                 << "," << zGlobal.y()                 << "," << zGlobal.z()                 << ")" << endl
	   << "\t x = t = (" << modules[i]->rotation().xx() << "," << modules[i]->rotation().yx() << "," << modules[i]->rotation().zx() << ")" << endl
	   << "\t [Rec] = (" << xGlobal.x()                 << "," << xGlobal.y()                 << "," << xGlobal.z()                 << ")" << endl
	   << "\t y = k = (" << modules[i]->rotation().xy() << "," << modules[i]->rotation().yy() << "," << modules[i]->rotation().zy() << ")" << endl
	   << "\t [Rec] = (" << yGlobal.x()                 << "," << yGlobal.y()                 << "," << yGlobal.z()                 << ")" << endl;
    
  }
  
  // params
  unsigned int chan_per_apv = 128;
  unsigned int apv_tib  = 
    6 * ( tib_L12_rphiN + tib_L12_sterN ) + 
    4 * tib_L34_rphiN                     ;
  unsigned int chan_tib = apv_tib*chan_per_apv;
  unsigned int apv_tid  = 
    6 * ( tid_r1_rphiN + tid_r1_sterN ) + 
    6 * ( tid_r2_rphiN + tid_r2_sterN ) + 
    4 * tid_r3_rphiN                    ;
  unsigned int chan_tid = apv_tid*chan_per_apv;
  unsigned int apv_tob  = 
    4 * ( tob_R12_rphiN + tob_R12_sterN ) + 
    4 * tob_R34_rphiN                     + 
    6 * tob_R56_rphiN                     ;
  unsigned int chan_tob = apv_tob*chan_per_apv;
  unsigned int apv_tec  = 
    6 * ( tec_r1_rphiN + tec_r1_sterN ) + 
    6 * ( tec_r2_rphiN + tec_r2_sterN ) + 
    4 * tec_r3_rphiN                    + 
    4 * tec_r4_rphiN                    + 
    6 * ( tec_r5_rphiN + tec_r5_sterN)  +
    4 * tec_r6_rphiN                    + 
    4 * tec_r7_rphiN                    ;
  unsigned int chan_tec = apv_tec*chan_per_apv;
  unsigned int apv_tot = apv_tib + apv_tid + apv_tob + apv_tec;
  unsigned int chan_tot = chan_tib + chan_tid + chan_tob + chan_tec;
  //
  
  // summary
  Output << "---------------------" << endl;
  Output << " Counters " << endl;
  Output << "---------------------" << endl;
  Output << " TIB    = " << tibN << endl;
  Output << "   L12 rphi   = " << tib_L12_rphiN << endl;
  Output << "   L12 stereo = " << tib_L12_sterN << endl;
  Output << "   L34        = " << tib_L34_rphiN << endl;
  Output << "   Active Silicon Detectors"             << endl;
  Output << "     Weight  = " << weight_tib << " kg" << endl;
  Output << "     Volume  = " << volume_tib << " cm3" << endl;
  Output << "     Surface = " << activeSurface_tib << " cm2" << endl;
  Output << "        APVs     = " << apv_tib << endl;;
  Output << "        channels = " << chan_tib << endl;;
  Output << " TID    = " << tidN << endl;
  Output << "   r1 rphi    = " << tid_r1_rphiN << endl;
  Output << "   r1 stereo  = " << tid_r1_sterN << endl;
  Output << "   r2 rphi    = " << tid_r2_rphiN << endl;
  Output << "   r2 stereo  = " << tid_r2_sterN << endl;
  Output << "   r3 rphi    = " << tid_r3_rphiN << endl;
  Output << "   Active Silicon Detectors"             << endl;
  Output << "     Weight  = " << weight_tid << " kg" << endl;
  Output << "     Volume  = " << volume_tid << " cm3" << endl;;
  Output << "     Surface = " << activeSurface_tid << " cm2" << endl;
  Output << "        APVs     = " << apv_tid << endl;;
  Output << "        channels = " << chan_tid << endl;;
  Output << " TOB    = " << tobN << endl;
  Output << "   R12 rphi   = " << tob_R12_rphiN << endl;
  Output << "   R12 stereo = " << tob_R12_sterN << endl;
  Output << "   R34        = " << tob_R34_rphiN << endl;
  Output << "   R56        = " << tob_R56_rphiN << endl;
  Output << "   Active Silicon Detectors"             << endl;
  Output << "     Weight  = " << weight_tob << " kg" << endl;
  Output << "     Volume  = " << volume_tob << " cm3" << endl;;
  Output << "     Surface = " << activeSurface_tob << " cm2" << endl;
  Output << "        APVs     = " << apv_tob << endl;;
  Output << "        channels = " << chan_tob << endl;;
  Output << " TEC    = " << tecN << endl;
  Output << "   r1 rphi    = " << tec_r1_rphiN << endl;
  Output << "   r1 stereo  = " << tec_r1_sterN << endl;
  Output << "   r2 rphi    = " << tec_r2_rphiN << endl;
  Output << "   r2 stereo  = " << tec_r2_sterN << endl;
  Output << "   r3 rphi    = " << tec_r3_rphiN << endl;
  Output << "   r4 rphi    = " << tec_r4_rphiN << endl;
  Output << "   r5 rphi    = " << tec_r5_rphiN << endl;
  Output << "   r5 stereo  = " << tec_r5_sterN << endl;
  Output << "   r6 rphi    = " << tec_r6_rphiN << endl;
  Output << "   r7 rphi    = " << tec_r7_rphiN << endl;
  Output << "   Active Silicon Detectors"             << endl;
  Output << "     Weight = " << weight_tec << " kg" << endl;
  Output << "     Volume = " << volume_tec << " cm3" << endl;;
  Output << "     Surface = " << activeSurface_tec << " cm2" << endl;
  Output << "        APVs     = " << apv_tec << endl;;
  Output << "        channels = " << chan_tec << endl;;
  Output << "---------------------" << endl;
  Output << " Total Weight      = " << weight_total << " kg" << endl;
  Output << " Total Volume      = " << volume_total << " cm3" << endl;;
  Output << " Total Active Area = " << activeSurface_total << " cm2" << endl;
  Output << "        APVs     = " << apv_tot << endl;;
  Output << "        channels = " << chan_tot << endl;;
  //
}

//define this as a plug-in
DEFINE_FWK_MODULE(ModuleInfo)
  
