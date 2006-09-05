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
//    uint32_t bits = geomid.rawId();
//      int b1 = (bits>>28);
//      int b2 = ((bits>>25)&0xF7);
//      int b3 = ((bits)&0xFFFFFF); 
//      int barrel = (bits>>25)&0x7;
//      int zet = (bits>>2)&0xF;
//      cout 
//           <<" bits  "<<*reinterpret_cast<const bitset<32>* >(&bits) << endl
//           <<" bits: "<<reinterpret_cast<const bitset<32>& >(b1)<<endl
//           <<" bits: "<<reinterpret_cast<const bitset<32>& >(b2)<<endl
//           <<" bits:         "<<reinterpret_cast<const bitset<24>& >(b3)<<endl;
//     cout      <<" part: "<<barrel<<" zet: "<<zet<<endl;
    PixelModuleName * name =  0;
    if (1 == geomid.subdetId()) {
      name = new PixelBarrelName(geomid);
//      const PixelTopology & tpl = pxUnit->specificTopology();
//      cout <<" NAME: "<<name->name()<<" rows, columns: "<< tpl.nrows()<<","<<tpl.ncolumns()<<endl;
//      LocalPoint local;
//      local = LocalPoint(0,0,0); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
//      local = LocalPoint(1,0,0); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
//      local = LocalPoint(0,1,0); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
//      local = LocalPoint(0,0,1); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
      
    } else {
      name = new PixelEndcapName(geomid);
//      const PixelTopology & tpl = pxUnit->specificTopology();
//      cout <<" NAME: "<<name->name()<<" rows, columns: "<< tpl.nrows()<<","<<tpl.ncolumns()<<endl;
//      LocalPoint local;
//      local = LocalPoint(0,0,0); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
//      local = LocalPoint(1,0,0); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
//      local = LocalPoint(0,1,0); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
//      local = LocalPoint(0,0,1); cout <<local<<"global: "<<(*pxUnit).toGlobal(local) <<endl;
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
