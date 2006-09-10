#include <iostream>

#include "CalibTracker/SiPixelConnectivity/interface/SiPixelFedCablingMapBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <ostream>
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

#include "CalibTracker/SiPixelConnectivity/interface/PixelToFEDAssociate.h"
#include "CalibTracker/SiPixelConnectivity/interface/TRange.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelBarrelLinkMaker.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelEndcapLinkMaker.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"

#include <bitset>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sipixelobjects;


SiPixelFedCablingMapBuilder::SiPixelFedCablingMapBuilder(const string & associatorName) : theAssociatorName(associatorName)
{ }

SiPixelFedCablingMap * SiPixelFedCablingMapBuilder::produce(
   const edm::EventSetup& setup)
{
   FEDNumbering fednum;
// TRange<int> fedIds = fednum.getSiPixelFEDIds();
   TRange<int> fedIds(0,40);
   edm::LogInfo("SiPixelFedCablingMapBuilder")<<"pixel fedid range: "<<fedIds;

   vector<FedSpec> fedSpecs(fedIds.max()-fedIds.min()+1); 
   for (int id=fedIds.first; id<=fedIds.second; id++) {
     FedSpec fs={ id, vector<PixelModuleName* >(), vector<uint32_t>()};
     int idx = id - fedIds.min();
     fedSpecs[idx]= fs;
   }

  edm::ESHandle<PixelToFEDAssociate> associator;
  setup.get<TrackerDigiGeometryRecord>().get(theAssociatorName,associator);
  const PixelToFEDAssociate & name2fed = *associator; 
  
  string version = name2fed.version();
  SiPixelFedCablingMap * result = new SiPixelFedCablingMap(version);


  LogDebug("read tracker geometry...");
  edm::ESHandle<TrackerGeometry> pDD;
  setup.get<TrackerDigiGeometryRecord>().get( pDD );
  LogDebug("tracker geometry read")<<"There are: "<<  pDD->dets().size() <<" detectors";

  typedef TrackerGeometry::DetContainer::const_iterator ITG;
  int npxdets = 0;
  for (ITG it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    const PixelGeomDetUnit * pxUnit = dynamic_cast<const PixelGeomDetUnit*>(*it);
    if (pxUnit  ==0 ) continue;
    npxdets++;
    DetId geomid = pxUnit->geographicalId();
    PixelModuleName * name =  0;
    if (1 == geomid.subdetId()) {
      name = new PixelBarrelName(geomid);
    } else {
      name = new PixelEndcapName(geomid);
//      cout << " NAME: "<<name->name()<<myprint(pxUnit)<<endl;
    } 
    int fedId = name2fed( *name);
    if ( fedIds.inside(fedId) ) {
      int idx = fedId - fedIds.min();
      fedSpecs[idx].rawids.push_back(geomid.rawId());
      fedSpecs[idx].names.push_back(name);
    } else edm::LogError("SiPixelFedCablingMapBuilder")
          <<"problem with numbering! "<<fedId<<" name: " << name->name();
  }
  LogDebug("tracker geometry read")<<"There are: "<< npxdets<<" pixel detetors";
  // construct FEDs
  typedef vector<FedSpec>::iterator FI;
  for ( FI it = fedSpecs.begin(); it != fedSpecs.end(); it++) {
    int fedId = it->fedId;
    vector<PixelModuleName* > names = it->names;
    vector<uint32_t> units = it->rawids;
    if ( names.size() == 0) continue;
    PixelFEDCabling fed(fedId);
    bool barrel = it->names.front()->isBarrel();
    if (barrel) {
      PixelFEDCabling::Links links = 
          PixelBarrelLinkMaker(&fed).links(names,units);
      fed.setLinks(links);
      result->addFed(fed);
    } else {
      PixelFEDCabling::Links links =
          PixelEndcapLinkMaker(&fed).links(names,units);
      fed.setLinks(links);
      result->addFed(fed);
    }
  }

  //clear names:
  for ( FI it = fedSpecs.begin(); it != fedSpecs.end(); it++) {
    vector<PixelModuleName* > names = it->names;
    typedef vector<PixelModuleName* >::const_iterator IN;
    for (IN name = names.begin(); name != names.end(); name++) delete (*name);
  } 
  return result;
}
std::string SiPixelFedCablingMapBuilder::myprint(const PixelGeomDetUnit * pxUnit)
{
  std::ostringstream str;
  const PixelTopology & tpl = pxUnit->specificTopology();
  LocalPoint local;
  GlobalPoint global;

  local = LocalPoint(0,0,0); global = (*pxUnit).toGlobal(local);
  float phi = 180*atan2(global.x(),global.y())/M_PI; if (phi > 180.) phi = 360-phi;
  float r = global.perp();
  float z = global.z();
  str <<"    GEOMETRY: "<<" r="<<r<<" phi="<<phi<<" z="<<z;
  str <<"   top:"<<tpl.nrows()<<","<<tpl.ncolumns();
      
//      local = LocalPoint(1,0,0); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
//      local = LocalPoint(0,1,0); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
//      local = LocalPoint(0,0,1); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
      
  return str.str();
}
