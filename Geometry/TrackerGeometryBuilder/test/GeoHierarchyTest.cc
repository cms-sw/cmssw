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
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerDebugNavigator.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "Geometry/TrackerNumberingBuilder/interface/CmsTrackerStringToEnum.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "DataFormats/Common/interface/Trie.h"


#include<string>
#include<iostream>



struct Print {
  typedef edm::TrieNode<const GeometricDet *> const node;
  void operator()(node & n, std::string const & label) const {
    if (!n.getValue()) return; 
    for (size_t i=0; i<label.size();++i)
      std::cout << int(label[i]) <<'/';
    std::cout << " " << n.getValue()->name().name() << std::endl;
  }
  
};



class GeoHierarchy : public edm::EDAnalyzer {
public:
  explicit GeoHierarchy( const edm::ParameterSet& );
  ~GeoHierarchy();
  
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  bool fromDDD_;
  bool printDDD_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
  static const double density_units = 6.24151e+18;

//
// constructors and destructor
//
GeoHierarchy::GeoHierarchy( const edm::ParameterSet& ps )
{
  fromDDD_ = ps.getParameter<bool>("fromDDD");
  printDDD_ = ps.getUntrackedParameter<bool>("printDDD", true);
 //now do what ever initialization is needed
  
}


GeoHierarchy::~GeoHierarchy()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
GeoHierarchy::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  edm::LogInfo("GeoHierarchy") << "begins";
  
  // build a trie of const GeometricDet
  typedef GeometricDet const * GDetP;
  edm::Trie<GDetP>	trie(0);
  
  typedef edm::TrieNode<GDetP> Node;
  typedef Node const * node_pointer; // sigh....
  typedef edm::TrieNodeIter<GDetP> node_iterator;
  
  
  
  //first instance tracking geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
  //
  GeometricDet const * rDD = pDD->trackerDet();
  int last=0;
  std::vector<const GeometricDet*> modules; (*rDD).deepComponents(modules);
  std::cout << "In Tracker Geom there are " << modules.size() 
	    << "modules" << std::endl; 
  try {
  for(unsigned int i=0; i<modules.size();i++){
    last = i;
    unsigned int rawid = modules[i]->geographicalID().rawId();
    int subdetid = modules[i]->geographicalID().subdetId();
    switch (subdetid) {
      
      // PXB
    case 1:
      {
	std::string name = modules[i]->name().name();
	PXBDetId module(rawid);
	char theLayer  = module.layer();
	char theLadder = module.ladder();
	char theModule = module.module();
	char key[] = { 1, theLayer , theLadder, theModule};
	trie.addEntry(key,4, modules[i]);
	
	
	break;
      }
      
      // PXF
    case 2:
      {
	std::string name = modules[i]->name().name();
	PXFDetId module(rawid);
	char thePanel  = module.panel();
	char theDisk   = module.disk();
	char theBlade  = module.blade();
	char theModule = module.module();
	char key[] = { 2,
		       char(module.side()),
		       thePanel , theDisk, 
		       theBlade, theModule};
	trie.addEntry(key,6, modules[i]);
	
	break;
      }
      
      // TIB
    case 3:
      {
	std::string name = modules[i]->name().name();
	TIBDetId module(rawid);
	char              theLayer  = module.layer();
	std::vector<unsigned int> theString = module.string();
	char             theModule = module.module();
	std::string side;
	std::string part;
	side = (theString[0] == 1 ) ? "-" : "+";
	part = (theString[1] == 1 ) ? "int" : "ext";
	char key[] = { 3, 
		       theLayer, 
		       char(theString[0]),
		       char(theString[1]), 
		       char(theString[2]), 
		       theModule,
		       char(module.stereo())
	};
	trie.addEntry(key,7, modules[i]);
	
	
	
	break;
      }
      
      // TID
    case 4:
      {
	std::string name = modules[i]->name().name();
	TIDDetId module(rawid);
	unsigned int         theDisk   = module.wheel();
	unsigned int         theRing   = module.ring();
	std::vector<unsigned int> theModule = module.module();
	std::string side;
	std::string part;
	side = (module.side() == 1 ) ? "-" : "+";
	part = (theModule[0] == 1 ) ? "back" : "front";
	char key[] = { 4, 
		       char(module.side()),
		       theDisk , 
		       theRing,
		       char(theModule[0]), 
		       char(theModule[1]),
		       char(module.stereo())
	};
	trie.addEntry(key,7, modules[i]);
	
	break;
      }
      
      // TOB
    case 5:
      {
	std::string name = modules[i]->name().name();
	TOBDetId module(rawid);
	unsigned int              theLayer  = module.layer();
	std::vector<unsigned int> theRod    = module.rod();
	unsigned int              theModule = module.module();
	std::string side;
	std::string part;
	side = (theRod[0] == 1 ) ? "-" : "+";
	char key[] = { 5, theLayer , 
		       char(theRod[0]), 
		       char(theRod[1]), 
		       theModule,
		       char(module.stereo())};
	trie.addEntry(key,6, modules[i]);
	
	break;
      }
      
      // TEC
    case 6:
      {
	std::string name = modules[i]->name().name();
	TECDetId module(rawid);
	unsigned int              theWheel  = module.wheel();
	unsigned int              theModule = module.module();
	std::vector<unsigned int> thePetal  = module.petal();
	unsigned int              theRing   = module.ring();
	std::string side;
	std::string petal;
	side  = (module.side() == 1 ) ? "-" : "+";
	petal = (thePetal[0] == 1 ) ? "back" : "front";
	// int out_side  = (module.side() == 1 ) ? -1 : 1;
	
	char key[] = { 6, 
		       char(module.side()),
		       theWheel,
		       char(thePetal[0]), 
		       char(thePetal[1]),
		       theRing,
		       theModule,
		       char(module.stereo())};
	trie.addEntry(key,8, modules[i]);
	
	/*
	unsigned int out_disk = module.wheel();
	unsigned int out_sector = thePetal[1];
	int out_petal = (thePetal[0] == 1 ) ? 1 : -1;
	// swap sector numbers for TEC-
	if (out_side == -1) {
	  // fine for back petals, substract 1 for front petals
	  if (out_petal == -1) {
	    out_sector = (out_sector+6) % 8 + 1;
	  }
	}
	unsigned int out_ring = module.ring();
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
	unsigned int out_module;
	if (out_ring == 1 || out_ring == 2 || out_ring == 5) {
	  // rings with stereo modules
	  // create number odd by default
	  out_module = 2*(module.module()-1)+1;
	  if (out_sensor == 1) {
	    // in even rings, stereo modules are the even ones
	    if (out_ring == 2)
	      out_module += 1;
	  }
	  else
	    // in odd rings, stereo modules are the odd ones
	    if (out_ring != 2)
	      out_module += 1;
	}
	else {
	  out_module = module.module();
	}
	*/
	break;
      }
    default:
      std::cerr << " WARNING no Silicon Strip detector, I got a " 
		<< rawid << " " << subdetid << std::endl;
    }
  }
  }
  catch(edm::VinException const & e) {
    std::cout << "in filling " << e.what() << std::endl;
    unsigned int rawid = modules[last]->geographicalID().rawId();
    int subdetid = modules[last]->geographicalID().subdetId();
    std::cout << rawid << " " << subdetid
	      << " " << modules[last]->name().name() << std::endl;
  }
    
  try {
    Print pr;
    edm::walkTrie(pr,*trie.getInitialNode());
    std::cout << std::endl; 
  }
  catch(edm::VinException const & e) {
    std::cout << "in walking " << e.what() << std::endl;
  }
    
}


  //define this as a plug-in
DEFINE_FWK_MODULE(GeoHierarchy);
  
