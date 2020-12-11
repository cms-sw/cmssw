#include <iostream>

#include "CalibTracker/SiPixelConnectivity/interface/SiPixelFedCablingMapBuilder.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include <ostream>
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"

#include "CalibTracker/SiPixelConnectivity/interface/PixelToFEDAssociate.h"
#include "CalibTracker/SiPixelConnectivity/interface/TRange.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelBarrelLinkMaker.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelEndcapLinkMaker.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelToLNKAssociateFromAscii.h"

#include <bitset>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sipixelobjects;

SiPixelFedCablingMapBuilder::SiPixelFedCablingMapBuilder(edm::ConsumesCollector&& iCC,
                                                         const string fileName,
                                                         const bool phase1)
    : fileName_(fileName)  //, phase1_(phase1) not used anymore
{
  trackerTopoToken_ = iCC.esConsumes<TrackerTopology, TrackerTopologyRcd>();
  trackerGeomToken_ = iCC.esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
}

SiPixelFedCablingTree* SiPixelFedCablingMapBuilder::produce(const edm::EventSetup& setup) {
  // Access geometry
  edm::LogInfo("read tracker geometry...");
  edm::ESHandle<TrackerGeometry> pDD = setup.getHandle(trackerGeomToken_);
  edm::LogInfo("tracker geometry read") << "There are: " << pDD->dets().size() << " detectors";

  // Test new TrackerGeometry features
  //cout << "Test of TrackerGeometry::isThere";
  //cout  << " is there PixelBarrel: " << pDD->isThere(GeomDetEnumerators::PixelBarrel);
  //cout  << " is there PixelEndcap: " << pDD->isThere(GeomDetEnumerators::PixelEndcap);
  //cout  << " is there P1PXB: " << pDD->isThere(GeomDetEnumerators::P1PXB);
  //cout  << " is there P1PXEC: " << pDD->isThere(GeomDetEnumerators::P1PXEC);
  //cout  << endl;

  // switch on the phase1
  if ((pDD->isThere(GeomDetEnumerators::P1PXB)) || (pDD->isThere(GeomDetEnumerators::P1PXEC))) {
    phase1_ = true;
    //cout<<" this is phase1 "<<endl;
    edm::LogInfo("SiPixelFedCablingMapBuilder") << " pixel phase1 setup ";
  } else {
    phase1_ = false;
    //cout<<" this is phase0 "<<endl;
    edm::LogInfo("SiPixelFedCablingMapBuilder") << " pixel phase0 setup ";
  }

  int MINFEDID = FEDNumbering::MINSiPixelFEDID;
  int MAXFEDID = FEDNumbering::MAXSiPixelFEDID;
  if (phase1_) {
    // bpix 1200-1239, fpix 1240-1255
    MINFEDID = FEDNumbering::MINSiPixeluTCAFEDID;
    MAXFEDID = FEDNumbering::MAXSiPixeluTCAFEDID;  // is actually 1349, might work
  }
  TRange<int> fedIds(MINFEDID, MAXFEDID);
  edm::LogInfo("SiPixelFedCablingMapBuilder") << "pixel fedid range: " << fedIds;

  // in the constrcuctor init() is called which reads the ascii file and makes
  // the map roc<->link
  // what is this junk? Replace by fixed associator.
  //edm::ESHandle<PixelToFEDAssociate> associator;
  //setup.get<TrackerDigiGeometryRecord>().get(theAssociatorName,associator);
  //PixelToFEDAssociate * associator = new PixelToLNKAssociateFromAscii("pixelToLNK.ascii",phase1_);
  PixelToFEDAssociate* associator = new PixelToLNKAssociateFromAscii(fileName_, phase1_);

  const PixelToFEDAssociate& name2fed = *associator;

  string version = name2fed.version();
  SiPixelFedCablingTree* result = new SiPixelFedCablingTree(version);
  edm::LogInfo(" version ") << version << endl;

  // Access topology
  edm::ESHandle<TrackerTopology> tTopo = setup.getHandle(trackerTopoToken_);
  const TrackerTopology* tt = tTopo.product();

  typedef TrackerGeometry::DetContainer::const_iterator ITG;
  int npxdets = 0;

  typedef std::vector<pair<PixelModuleName*, uint32_t> > UNITS;
  UNITS units;

  for (ITG it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
    const PixelGeomDetUnit* pxUnit = dynamic_cast<const PixelGeomDetUnit*>(*it);
    if (pxUnit == nullptr)
      continue;
    npxdets++;
    DetId geomid = pxUnit->geographicalId();
    PixelModuleName* name = nullptr;
    if (1 == geomid.subdetId()) {  // bpix
      name = new PixelBarrelName(geomid, tt, phase1_);
    } else {  // fpix
      name = new PixelEndcapName(geomid, tt, phase1_);
    }
    edm::LogInfo(" NAME: ") << name->name();
    //cout << " NAME: "<<name->name()<<" "<<geomid.rawId()<<
    //" "<<myprint(pxUnit)<<endl;
    units.push_back(std::make_pair(name, geomid.rawId()));
  }

  // This produces  a simple, unrealistic map, NOT USED ANYMORE
  // if (theAssociatorName=="PixelToFEDAssociateFromAscii") {
  //   cout <<" HERE PixelToFEDAssociateFromAscii"<<endl;
  //   vector<FedSpec> fedSpecs(fedIds.max()-fedIds.min()+1);
  //   for (int id=fedIds.first; id<=fedIds.second; id++) {
  //     FedSpec fs={ id, vector<PixelModuleName* >(), vector<uint32_t>()};
  //     int idx = id - fedIds.min();
  //     fedSpecs[idx]= fs;
  //   }
  //   for (UNITS::iterator iu=units.begin(); iu != units.end(); iu++) {
  //     PixelModuleName* name = (*iu).first;
  //     uint32_t rawId = (*iu).second;
  //     int fedId = name2fed( *name);
  //     if ( fedIds.inside(fedId) ) {
  // 	int idx = fedId - fedIds.min();
  // 	fedSpecs[idx].rawids.push_back(rawId);
  // 	fedSpecs[idx].names.push_back(name);
  //     } else edm::LogError("SiPixelFedCablingMapBuilder")
  // 	       <<"problem with numbering! "<<fedId<<" name: " << name->name();
  //   }
  //   edm::LogInfo("tracker geometry read")<<"There are: "<< npxdets<<" pixel detetors";
  //   // construct FEDs
  //   typedef vector<FedSpec>::iterator FI;
  //   for ( FI it = fedSpecs.begin(); it != fedSpecs.end(); it++) {
  //     int fedId = it->fedId;
  //     vector<PixelModuleName* > names = it->names;
  //     vector<uint32_t> units = it->rawids;
  //     if ( names.size() == 0) continue;
  //     PixelFEDCabling fed(fedId);
  //     bool barrel = it->names.front()->isBarrel();
  //     if (barrel) {
  // 	PixelFEDCabling::Links links =
  //         PixelBarrelLinkMaker(&fed).links(names,units);
  // 	fed.setLinks(links);
  // 	result->addFed(fed);
  //     } else {
  // 	PixelFEDCabling::Links links =
  //         PixelEndcapLinkMaker(&fed).links(names,units);
  // 	fed.setLinks(links);
  // 	result->addFed(fed);
  //     }
  //   }
  // } else {     // This is what is really used

  PixelToFEDAssociate::DetectorRocId detectorRocId;
  edm::LogInfo(" HERE PixelToLNKAssociateFromAscii");
  for (UNITS::iterator iu = units.begin(); iu != units.end(); iu++) {
    PixelModuleName* name = (*iu).first;
    detectorRocId.module = name;
    //for (int rocDetId=0; rocDetId<=16; rocDetId++) {
    for (int rocDetId = 0; rocDetId < 16; rocDetId++) {
      detectorRocId.rocDetId = rocDetId;
      const PixelToFEDAssociate::CablingRocId* cablingRocId = name2fed(detectorRocId);
      if (cablingRocId) {
        sipixelobjects::PixelROC roc(iu->second, rocDetId, cablingRocId->rocLinkId);
        result->addItem(cablingRocId->fedId, cablingRocId->linkId, roc);
        edm::LogInfo(" ok ") << name->name() << " " << rocDetId << " " << cablingRocId->fedId << " "
                             << cablingRocId->linkId;
      } else {  // did it fail?
        edm::LogInfo(" failed ") << name->name() << " " << rocDetId;
        //cout<<" failed "<<name->name()<<" "<<rocDetId<<endl;
      }
    }
  }
  //}

  //clear names:
  for (UNITS::iterator iu = units.begin(); iu != units.end(); iu++)
    delete iu->first;

  return result;
}
std::string SiPixelFedCablingMapBuilder::myprint(const PixelGeomDetUnit* pxUnit) {
  std::ostringstream str;
  const PixelTopology& tpl = pxUnit->specificTopology();
  LocalPoint local;
  GlobalPoint global;

  local = LocalPoint(0, 0, 0);
  global = (*pxUnit).toGlobal(local);
  // phi measured from Y axis
  float phi = 180 * atan2(global.x(), global.y()) / M_PI;
  if (phi > 180.)
    phi = phi - 360;
  float r = global.perp();
  float z = global.z();
  str << "    POSITION: "
      << " r=" << r << " phi=" << phi << " z=" << z;
  str << "   (rows,coll:" << tpl.nrows() << "," << tpl.ncolumns() << ")";
  str << endl;
  local = LocalPoint(0, 0, 0);
  str << local << "global: " << (*pxUnit).toGlobal(local) << endl;
  local = LocalPoint(1, 0, 0);
  str << local << "global: " << (*pxUnit).toGlobal(local) << endl;
  local = LocalPoint(0, 1, 0);
  str << local << "global: " << (*pxUnit).toGlobal(local) << endl;
  local = LocalPoint(0, 0, 1);
  str << local << "global: " << (*pxUnit).toGlobal(local) << endl;

  return str.str();
}
