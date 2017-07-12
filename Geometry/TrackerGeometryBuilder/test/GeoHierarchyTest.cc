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
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/trackerHierarchy.h"

#include "DataFormats/Common/interface/Trie.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"


#include<string>
#include<iostream>

template<typename Det>
struct Print {
  // typedef edm::TrieNode<Det> const node;
  void operator()(Det det, std::string const & label) const {
    if (!det) return; 
    for (char i : label)
      std::cout << int(i) <<'/';
    std::cout << " " << det->geographicalId().rawId() << std::endl;
  }
  
};

class GeoHierarchy : public edm::one::EDAnalyzer<> {
public:
  explicit GeoHierarchy( const edm::ParameterSet& );
  ~GeoHierarchy() override;
  
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  // ----------member data ---------------------------
  bool fromDDD_;
  bool printDDD_;
};

GeoHierarchy::GeoHierarchy( const edm::ParameterSet& ps )
{
  fromDDD_ = ps.getParameter<bool>("fromDDD");
  printDDD_ = ps.getUntrackedParameter<bool>("printDDD", true);
}

GeoHierarchy::~GeoHierarchy()
{}

template<typename Iter>
void constructAndDumpTrie(const TrackerTopology* tTopo, Iter b, Iter e) {
  typedef typename std::iterator_traits<Iter>::value_type Det;
  edm::Trie<Det> trie(0);
  typedef edm::TrieNode<Det> Node;
  typedef Node const * node_pointer; // sigh....
  typedef edm::TrieNodeIter<Det> node_iterator;

  std::cout << "In Tracker Geom there are " << e-b 
	    << " modules" << std::endl; 
  Iter last=b;
  try {
    for(;b!=e; ++b) {
      last = b;
      unsigned int rawid = (*b)->geographicalId().rawId();
      trie.insert(trackerHierarchy(tTopo, rawid), *b); 
    }
  }
  catch(edm::Exception const & ex) {
    std::cout << "in filling " << ex.what() << std::endl;
    unsigned int rawid = (*last)->geographicalId().rawId();
    int subdetid = (*last)->geographicalId().subdetId();
    std::cout << rawid << " " << subdetid << std::endl;
  }
  
  try {
    Print<Det> pr;
    edm::walkTrie(pr,*trie.initialNode());
    std::cout << std::endl; 


    unsigned int layerId[] = {1,3,5,21,22,41,42,61,62};
    int layerSize[9];
    for (int i=0;i<9;i++) {
      std::string s;
      if (layerId[i]>9) s+=char(layerId[i]/10); 
      s+=char(layerId[i]%10);
      node_iterator eit;	
      node_iterator p(trie.node(s));
      layerSize[i] = std::distance(p,eit);
    }

    edm::LogInfo("TkDetLayers") 
      << "------ Geometry constructed with: ------" << "\n"
      << "n pxlBarLayers: " <<  layerSize[0] << "\n"
      << "n tibLayers:    " << layerSize[1] << "\n"
      << "n tobLayers:    " << layerSize[2] << "\n"
      << "n negPxlFwdLayers: " << layerSize[3] << "\n"
      << "n posPxlFwdLayers: " << layerSize[4] << "\n"
      << "n negTidLayers: " << layerSize[5] << "\n"
      << "n posTidLayers: " << layerSize[6] << "\n"
      << "n negTecLayers: " << layerSize[7] << "\n"
      << "n posTecLayers: " << layerSize[8] << "\n";
      
      //      << "n barreLayers:  " << this->barrelLayers().size() << "\n"
      //<< "n negforwardLayers: " << this->negForwardLayers().size() << "\n"
      //<< "n posForwardLayers: " << this->posForwardLayers().size() ;
  }
  catch(edm::Exception const & ex) {
    std::cout << "in walking " << ex.what() << std::endl;
  }
  
}

// ------------ method called to produce the data  ------------
void
GeoHierarchy::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  edm::LogInfo("GeoHierarchy") << "begins";
  
  //first instance tracking geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);
  edm::ESHandle<TrackerTopology> tTopo_handle;
  iSetup.get<TrackerTopologyRcd>().get(tTopo_handle);
  const TrackerTopology* tTopo = tTopo_handle.product();
  //
  GeometricDet const * rDD = pDD->trackerDet();
  std::vector<const GeometricDet*> modules; 
  (*rDD).deepComponents(modules);
  
  std::cout << "\nGeometricDet Hierarchy\n" << std::endl;
  constructAndDumpTrie(tTopo, modules.begin(),modules.end());
  
  std::cout << "\nGDet Hierarchy\n" << std::endl;
  constructAndDumpTrie(tTopo, pDD->dets().begin(),pDD->dets().end());
  




}

//define this as a plug-in
DEFINE_FWK_MODULE(GeoHierarchy);

