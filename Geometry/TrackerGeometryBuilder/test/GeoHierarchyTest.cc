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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"



#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/trackerHierarchy.h"

#include "DataFormats/Common/interface/Trie.h"


#include<string>
#include<iostream>



struct Print {
  typedef edm::TrieNode<const GeometricDet *> const node;
  void operator()(node & n, std::string const & label) const {
    if (!n.value()) return; 
    for (size_t i=0; i<label.size();++i)
      std::cout << int(label[i]) <<'/';
    std::cout << " " << n.value()->name().name() << std::endl;
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
      trie.insert(trackerHierarchy(rawid), modules[i]);
      
    }
  }
  catch(edm::Exception const & e) {
    std::cout << "in filling " << e.what() << std::endl;
    unsigned int rawid = modules[last]->geographicalID().rawId();
    int subdetid = modules[last]->geographicalID().subdetId();
    std::cout << rawid << " " << subdetid
	      << " " << modules[last]->name().name() << std::endl;
  }
  
  try {
    Print pr;
    edm::walkTrie(pr,*trie.initialNode());
    std::cout << std::endl; 
  }
  catch(edm::Exception const & e) {
    std::cout << "in walking " << e.what() << std::endl;
  }
    
}


  //define this as a plug-in
DEFINE_FWK_MODULE(GeoHierarchy);
  
