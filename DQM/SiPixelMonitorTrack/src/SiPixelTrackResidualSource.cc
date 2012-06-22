// Package:    SiPixelMonitorTrack
// Class:      SiPixelTrackResidualSource
// 
// class SiPixelTrackResidualSource SiPixelTrackResidualSource.cc 
//       DQM/SiPixelMonitorTrack/src/SiPixelTrackResidualSource.cc
//
// Description: SiPixel hit-to-track residual data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step 
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
//         Updated by Lukas Wehrli (plots for clusters on/off track added)
// $Id: SiPixelTrackResidualSource.cc,v 1.22 2012/01/19 11:26:26 merkelp Exp $


#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQM/SiPixelMonitorTrack/interface/SiPixelTrackResidualSource.h"


using namespace std;
using namespace edm;


SiPixelTrackResidualSource::SiPixelTrackResidualSource(const edm::ParameterSet& pSet) :
  pSet_(pSet),
  modOn( pSet.getUntrackedParameter<bool>("modOn",true) ),
  reducedSet( pSet.getUntrackedParameter<bool>("reducedSet",true) ),
  ladOn( pSet.getUntrackedParameter<bool>("ladOn",false) ), 
  layOn( pSet.getUntrackedParameter<bool>("layOn",false) ), 
  phiOn( pSet.getUntrackedParameter<bool>("phiOn",false) ), 
  ringOn( pSet.getUntrackedParameter<bool>("ringOn",false) ), 
  bladeOn( pSet.getUntrackedParameter<bool>("bladeOn",false) ), 
  diskOn( pSet.getUntrackedParameter<bool>("diskOn",false) )
 { 
   pSet_ = pSet; 
  debug_ = pSet_.getUntrackedParameter<bool>("debug", false); 
    src_ = pSet_.getParameter<edm::InputTag>("src"); 
    clustersrc_ = pSet_.getParameter<edm::InputTag>("clustersrc");
    tracksrc_ = pSet_.getParameter<edm::InputTag>("trajectoryInput");
    dbe_ = edm::Service<DQMStore>().operator->();

  LogInfo("PixelDQM") << "SiPixelTrackResidualSource constructor" << endl;
  LogInfo ("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" 
            << layOn << "/" << phiOn << std::endl;
  LogInfo ("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" 
            << ringOn << std::endl;
}


SiPixelTrackResidualSource::~SiPixelTrackResidualSource() {
  LogInfo("PixelDQM") << "SiPixelTrackResidualSource destructor" << endl;

  std::map<uint32_t,SiPixelTrackResidualModule*>::iterator struct_iter;
  for (struct_iter = theSiPixelStructure.begin() ; struct_iter != theSiPixelStructure.end() ; struct_iter++){
    delete struct_iter->second;
    struct_iter->second = 0;
  }
}

void SiPixelTrackResidualSource::beginJob() {
  LogInfo("PixelDQM") << "SiPixelTrackResidualSource beginJob()" << endl;
  firstRun = true;
  NTotal=0;
  NLowProb=0;
}


void SiPixelTrackResidualSource::beginRun(const edm::Run& r, edm::EventSetup const& iSetup) {
  LogInfo("PixelDQM") << "SiPixelTrackResidualSource beginRun()" << endl;

  if(firstRun){
  // retrieve TrackerGeometry for pixel dets
  edm::ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  if (debug_) LogVerbatim("PixelDQM") << "TrackerGeometry "<< &(*TG) <<" size is "<< TG->dets().size() << endl;
 
  // build theSiPixelStructure with the pixel barrel and endcap dets from TrackerGeometry
  for (TrackerGeometry::DetContainer::const_iterator pxb = TG->detsPXB().begin();  
       pxb!=TG->detsPXB().end(); pxb++) {
    if (dynamic_cast<PixelGeomDetUnit*>((*pxb))!=0) {
      SiPixelTrackResidualModule* module = new SiPixelTrackResidualModule((*pxb)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelTrackResidualModule*>((*pxb)->geographicalId().rawId(), module));
    }
  }
  for (TrackerGeometry::DetContainer::const_iterator pxf = TG->detsPXF().begin(); 
       pxf!=TG->detsPXF().end(); pxf++) {
    if (dynamic_cast<PixelGeomDetUnit*>((*pxf))!=0) {
      SiPixelTrackResidualModule* module = new SiPixelTrackResidualModule((*pxf)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelTrackResidualModule*>((*pxf)->geographicalId().rawId(), module));
    }
  }
  LogInfo("PixelDQM") << "SiPixelStructure size is " << theSiPixelStructure.size() << endl;

  // book residual histograms in theSiPixelFolder - one (x,y) pair of histograms per det
  SiPixelFolderOrganizer theSiPixelFolder;
  for (std::map<uint32_t, SiPixelTrackResidualModule*>::iterator pxd = theSiPixelStructure.begin(); 
       pxd!=theSiPixelStructure.end(); pxd++) {

    if(modOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first)) (*pxd).second->book(pSet_,reducedSet);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource Folder Creation Failed! "; 
    }
    if(ladOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,1)) {
	
	(*pxd).second->book(pSet_,reducedSet,1);
      }
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource ladder Folder Creation Failed! "; 
    }
    if(layOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,2)) (*pxd).second->book(pSet_,reducedSet,2);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource layer Folder Creation Failed! "; 
    }
    if(phiOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,3)) (*pxd).second->book(pSet_,reducedSet,3);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource phi Folder Creation Failed! "; 
    }
    if(bladeOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,4)) (*pxd).second->book(pSet_,reducedSet,4);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource Blade Folder Creation Failed! "; 
    }
    if(diskOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,5)) (*pxd).second->book(pSet_,reducedSet,5);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource Disk Folder Creation Failed! "; 
    }
    if(ringOn){
      if (theSiPixelFolder.setModuleFolder((*pxd).first,6)) (*pxd).second->book(pSet_,reducedSet,6);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource Ring Folder Creation Failed! "; 
    }
  }


//   edm::InputTag tracksrc = pSet_.getParameter<edm::InputTag>("trajectoryInput");
//   edm::InputTag clustersrc = pSet_.getParameter<edm::InputTag>("clustersrc");

  //number of tracks
  dbe_->setCurrentFolder("Pixel/Tracks");
  meNofTracks_ = dbe_->book1D("ntracks_" + tracksrc_.label(),"Number of Tracks",4,0,4);
  meNofTracks_->setAxisTitle("Number of Tracks",1);
  meNofTracks_->setBinLabel(1,"All");
  meNofTracks_->setBinLabel(2,"Pixel");
  meNofTracks_->setBinLabel(3,"BPix");
  meNofTracks_->setBinLabel(4,"FPix");

  //number of tracks in pixel fiducial volume
  dbe_->setCurrentFolder("Pixel/Tracks");
  meNofTracksInPixVol_ = dbe_->book1D("ntracksInPixVol_" + tracksrc_.label(),"Number of Tracks crossing Pixel fiducial Volume",2,0,2);
  meNofTracksInPixVol_->setAxisTitle("Number of Tracks",1);
  meNofTracksInPixVol_->setBinLabel(1,"With Hits");
  meNofTracksInPixVol_->setBinLabel(2,"Without Hits");

  //number of clusters (associated to track / not associated)
  dbe_->setCurrentFolder("Pixel/Clusters/OnTrack");
  meNofClustersOnTrack_ = dbe_->book1D("nclusters_" + clustersrc_.label() + "_tot","Number of Clusters (on track)",3,0,3);
  meNofClustersOnTrack_->setAxisTitle("Number of Clusters on Track",1);
  meNofClustersOnTrack_->setBinLabel(1,"All");
  meNofClustersOnTrack_->setBinLabel(2,"BPix");
  meNofClustersOnTrack_->setBinLabel(3,"FPix");
  dbe_->setCurrentFolder("Pixel/Clusters/OffTrack");
  meNofClustersNotOnTrack_ = dbe_->book1D("nclusters_" + clustersrc_.label() + "_tot","Number of Clusters (off track)",3,0,3);
  meNofClustersNotOnTrack_->setAxisTitle("Number of Clusters off Track",1);
  meNofClustersNotOnTrack_->setBinLabel(1,"All");
  meNofClustersNotOnTrack_->setBinLabel(2,"BPix");
  meNofClustersNotOnTrack_->setBinLabel(3,"FPix");

  //cluster charge and size
  //charge
  //on track
  dbe_->setCurrentFolder("Pixel/Clusters/OnTrack");
  meClChargeOnTrack_all = dbe_->book1D("charge_" + clustersrc_.label(),"Charge (on track)",500,0.,500.);
  meClChargeOnTrack_all->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_bpix = dbe_->book1D("charge_" + clustersrc_.label() + "_Barrel","Charge (on track, barrel)",500,0.,500.);
  meClChargeOnTrack_bpix->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_fpix = dbe_->book1D("charge_" + clustersrc_.label() + "_Endcap","Charge (on track, endcap)",500,0.,500.);
  meClChargeOnTrack_fpix->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_layer1 = dbe_->book1D("charge_" + clustersrc_.label() + "_Layer_1","Charge (on track, layer1)",500,0.,500.);
  meClChargeOnTrack_layer1->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_layer2 = dbe_->book1D("charge_" + clustersrc_.label() + "_Layer_2","Charge (on track, layer2)",500,0.,500.);
  meClChargeOnTrack_layer2->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_layer3 = dbe_->book1D("charge_" + clustersrc_.label() + "_Layer_3","Charge (on track, layer3)",500,0.,500.);
  meClChargeOnTrack_layer3->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_diskp1 = dbe_->book1D("charge_" + clustersrc_.label() + "_Disk_p1","Charge (on track, diskp1)",500,0.,500.);
  meClChargeOnTrack_diskp1->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_diskp2 = dbe_->book1D("charge_" + clustersrc_.label() + "_Disk_p2","Charge (on track, diskp2)",500,0.,500.);
  meClChargeOnTrack_diskp2->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_diskm1 = dbe_->book1D("charge_" + clustersrc_.label() + "_Disk_m1","Charge (on track, diskm1)",500,0.,500.);
  meClChargeOnTrack_diskm1->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_diskm2 = dbe_->book1D("charge_" + clustersrc_.label() + "_Disk_m2","Charge (on track, diskm2)",500,0.,500.);
  meClChargeOnTrack_diskm2->setAxisTitle("Charge size (in ke)",1);
  //off track
  dbe_->setCurrentFolder("Pixel/Clusters/OffTrack");
  meClChargeNotOnTrack_all = dbe_->book1D("charge_" + clustersrc_.label(),"Charge (off track)",500,0.,500.);
  meClChargeNotOnTrack_all->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_bpix = dbe_->book1D("charge_" + clustersrc_.label() + "_Barrel","Charge (off track, barrel)",500,0.,500.);
  meClChargeNotOnTrack_bpix->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_fpix = dbe_->book1D("charge_" + clustersrc_.label() + "_Endcap","Charge (off track, endcap)",500,0.,500.);
  meClChargeNotOnTrack_fpix->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_layer1 = dbe_->book1D("charge_" + clustersrc_.label() + "_Layer_1","Charge (off track, layer1)",500,0.,500.);
  meClChargeNotOnTrack_layer1->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_layer2 = dbe_->book1D("charge_" + clustersrc_.label() + "_Layer_2","Charge (off track, layer2)",500,0.,500.);
  meClChargeNotOnTrack_layer2->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_layer3 = dbe_->book1D("charge_" + clustersrc_.label() + "_Layer_3","Charge (off track, layer3)",500,0.,500.);
  meClChargeNotOnTrack_layer3->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_diskp1 = dbe_->book1D("charge_" + clustersrc_.label() + "_Disk_p1","Charge (off track, diskp1)",500,0.,500.);
  meClChargeNotOnTrack_diskp1->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_diskp2 = dbe_->book1D("charge_" + clustersrc_.label() + "_Disk_p2","Charge (off track, diskp2)",500,0.,500.);
  meClChargeNotOnTrack_diskp2->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_diskm1 = dbe_->book1D("charge_" + clustersrc_.label() + "_Disk_m1","Charge (off track, diskm1)",500,0.,500.);
  meClChargeNotOnTrack_diskm1->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_diskm2 = dbe_->book1D("charge_" + clustersrc_.label() + "_Disk_m2","Charge (off track, diskm2)",500,0.,500.);
  meClChargeNotOnTrack_diskm2->setAxisTitle("Charge size (in ke)",1);

  //size
  //on track
  dbe_->setCurrentFolder("Pixel/Clusters/OnTrack");
  meClSizeOnTrack_all = dbe_->book1D("size_" + clustersrc_.label(),"Size (on track)",100,0.,100.);
  meClSizeOnTrack_all->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_bpix = dbe_->book1D("size_" + clustersrc_.label() + "_Barrel","Size (on track, barrel)",100,0.,100.);
  meClSizeOnTrack_bpix->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_fpix = dbe_->book1D("size_" + clustersrc_.label() + "_Endcap","Size (on track, endcap)",100,0.,100.);
  meClSizeOnTrack_fpix->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_layer1 = dbe_->book1D("size_" + clustersrc_.label() + "_Layer_1","Size (on track, layer1)",100,0.,100.);
  meClSizeOnTrack_layer1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_layer2 = dbe_->book1D("size_" + clustersrc_.label() + "_Layer_2","Size (on track, layer2)",100,0.,100.);
  meClSizeOnTrack_layer2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_layer3 = dbe_->book1D("size_" + clustersrc_.label() + "_Layer_3","Size (on track, layer3)",100,0.,100.);
  meClSizeOnTrack_layer3->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_diskp1 = dbe_->book1D("size_" + clustersrc_.label() + "_Disk_p1","Size (on track, diskp1)",100,0.,100.);
  meClSizeOnTrack_diskp1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_diskp2 = dbe_->book1D("size_" + clustersrc_.label() + "_Disk_p2","Size (on track, diskp2)",100,0.,100.);
  meClSizeOnTrack_diskp2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_diskm1 = dbe_->book1D("size_" + clustersrc_.label() + "_Disk_m1","Size (on track, diskm1)",100,0.,100.);
  meClSizeOnTrack_diskm1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_diskm2 = dbe_->book1D("size_" + clustersrc_.label() + "_Disk_m2","Size (on track, diskm2)",100,0.,100.);
  meClSizeOnTrack_diskm2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXOnTrack_all = dbe_->book1D("sizeX_" + clustersrc_.label(),"SizeX (on track)",100,0.,100.);
  meClSizeXOnTrack_all->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXOnTrack_bpix = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Barrel","SizeX (on track, barrel)",100,0.,100.);
  meClSizeXOnTrack_bpix->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXOnTrack_fpix = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Endcap","SizeX (on track, endcap)",100,0.,100.);
  meClSizeXOnTrack_fpix->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXOnTrack_layer1 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Layer_1","SizeX (on track, layer1)",100,0.,100.);
  meClSizeXOnTrack_layer1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXOnTrack_layer2 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Layer_2","SizeX (on track, layer2)",100,0.,100.);
  meClSizeXOnTrack_layer2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXOnTrack_layer3 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Layer_3","SizeX (on track, layer3)",100,0.,100.);
  meClSizeXOnTrack_layer3->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXOnTrack_diskp1 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Disk_p1","SizeX (on track, diskp1)",100,0.,100.);
  meClSizeXOnTrack_diskp1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXOnTrack_diskp2 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Disk_p2","SizeX (on track, diskp2)",100,0.,100.);
  meClSizeXOnTrack_diskp2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXOnTrack_diskm1 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Disk_m1","SizeX (on track, diskm1)",100,0.,100.);
  meClSizeXOnTrack_diskm1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXOnTrack_diskm2 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Disk_m2","SizeX (on track, diskm2)",100,0.,100.);
  meClSizeXOnTrack_diskm2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYOnTrack_all = dbe_->book1D("sizeY_" + clustersrc_.label(),"SizeY (on track)",100,0.,100.);
  meClSizeYOnTrack_all->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYOnTrack_bpix = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Barrel","SizeY (on track, barrel)",100,0.,100.);
  meClSizeYOnTrack_bpix->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYOnTrack_fpix = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Endcap","SizeY (on track, endcap)",100,0.,100.);
  meClSizeYOnTrack_fpix->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYOnTrack_layer1 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Layer_1","SizeY (on track, layer1)",100,0.,100.);
  meClSizeYOnTrack_layer1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYOnTrack_layer2 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Layer_2","SizeY (on track, layer2)",100,0.,100.);
  meClSizeYOnTrack_layer2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYOnTrack_layer3 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Layer_3","SizeY (on track, layer3)",100,0.,100.);
  meClSizeYOnTrack_layer3->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYOnTrack_diskp1 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Disk_p1","SizeY (on track, diskp1)",100,0.,100.);
  meClSizeYOnTrack_diskp1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYOnTrack_diskp2 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Disk_p2","SizeY (on track, diskp2)",100,0.,100.);
  meClSizeYOnTrack_diskp2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYOnTrack_diskm1 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Disk_m1","SizeY (on track, diskm1)",100,0.,100.);
  meClSizeYOnTrack_diskm1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYOnTrack_diskm2 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Disk_m2","SizeY (on track, diskm2)",100,0.,100.);
  meClSizeYOnTrack_diskm2->setAxisTitle("Cluster size (in pixels)",1);
  //off track
  dbe_->setCurrentFolder("Pixel/Clusters/OffTrack");
  meClSizeNotOnTrack_all = dbe_->book1D("size_" + clustersrc_.label(),"Size (off track)",100,0.,100.);
  meClSizeNotOnTrack_all->setAxisTitle("Cluster size (in pixels)",1); 
  meClSizeNotOnTrack_bpix = dbe_->book1D("size_" + clustersrc_.label() + "_Barrel","Size (off track, barrel)",100,0.,100.);
  meClSizeNotOnTrack_bpix->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeNotOnTrack_fpix = dbe_->book1D("size_" + clustersrc_.label() + "_Endcap","Size (off track, endcap)",100,0.,100.);
  meClSizeNotOnTrack_fpix->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeNotOnTrack_layer1 = dbe_->book1D("size_" + clustersrc_.label() + "_Layer_1","Size (off track, layer1)",100,0.,100.);
  meClSizeNotOnTrack_layer1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeNotOnTrack_layer2 = dbe_->book1D("size_" + clustersrc_.label() + "_Layer_2","Size (off track, layer2)",100,0.,100.);
  meClSizeNotOnTrack_layer2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeNotOnTrack_layer3 = dbe_->book1D("size_" + clustersrc_.label() + "_Layer_3","Size (off track, layer3)",100,0.,100.);
  meClSizeNotOnTrack_layer3->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeNotOnTrack_diskp1 = dbe_->book1D("size_" + clustersrc_.label() + "_Disk_p1","Size (off track, diskp1)",100,0.,100.);
  meClSizeNotOnTrack_diskp1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeNotOnTrack_diskp2 = dbe_->book1D("size_" + clustersrc_.label() + "_Disk_p2","Size (off track, diskp2)",100,0.,100.);
  meClSizeNotOnTrack_diskp2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeNotOnTrack_diskm1 = dbe_->book1D("size_" + clustersrc_.label() + "_Disk_m1","Size (off track, diskm1)",100,0.,100.);
  meClSizeNotOnTrack_diskm1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeNotOnTrack_diskm2 = dbe_->book1D("size_" + clustersrc_.label() + "_Disk_m2","Size (off track, diskm2)",100,0.,100.);
  meClSizeNotOnTrack_diskm2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXNotOnTrack_all = dbe_->book1D("sizeX_" + clustersrc_.label(),"SizeX (off track)",100,0.,100.);
  meClSizeXNotOnTrack_all->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXNotOnTrack_bpix = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Barrel","SizeX (off track, barrel)",100,0.,100.);
  meClSizeXNotOnTrack_bpix->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXNotOnTrack_fpix = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Endcap","SizeX (off track, endcap)",100,0.,100.);
  meClSizeXNotOnTrack_fpix->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXNotOnTrack_layer1 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Layer_1","SizeX (off track, layer1)",100,0.,100.);
  meClSizeXNotOnTrack_layer1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXNotOnTrack_layer2 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Layer_2","SizeX (off track, layer2)",100,0.,100.);
  meClSizeXNotOnTrack_layer2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXNotOnTrack_layer3 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Layer_3","SizeX (off track, layer3)",100,0.,100.);
  meClSizeXNotOnTrack_layer3->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXNotOnTrack_diskp1 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Disk_p1","SizeX (off track, diskp1)",100,0.,100.);
  meClSizeXNotOnTrack_diskp1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXNotOnTrack_diskp2 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Disk_p2","SizeX (off track, diskp2)",100,0.,100.);
  meClSizeXNotOnTrack_diskp2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXNotOnTrack_diskm1 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Disk_m1","SizeX (off track, diskm1)",100,0.,100.);
  meClSizeXNotOnTrack_diskm1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeXNotOnTrack_diskm2 = dbe_->book1D("sizeX_" + clustersrc_.label() + "_Disk_m2","SizeX (off track, diskm2)",100,0.,100.);
  meClSizeXNotOnTrack_diskm2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYNotOnTrack_all = dbe_->book1D("sizeY_" + clustersrc_.label(),"SizeY (off track)",100,0.,100.);
  meClSizeYNotOnTrack_all->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYNotOnTrack_bpix = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Barrel","SizeY (off track, barrel)",100,0.,100.);
  meClSizeYNotOnTrack_bpix->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYNotOnTrack_fpix = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Endcap","SizeY (off track, endcap)",100,0.,100.);
  meClSizeYNotOnTrack_fpix->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYNotOnTrack_layer1 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Layer_1","SizeY (off track, layer1)",100,0.,100.);
  meClSizeYNotOnTrack_layer1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYNotOnTrack_layer2 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Layer_2","SizeY (off track, layer2)",100,0.,100.);
  meClSizeYNotOnTrack_layer2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYNotOnTrack_layer3 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Layer_3","SizeY (off track, layer3)",100,0.,100.);
  meClSizeYNotOnTrack_layer3->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYNotOnTrack_diskp1 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Disk_p1","SizeY (off track, diskp1)",100,0.,100.);
  meClSizeYNotOnTrack_diskp1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYNotOnTrack_diskp2 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Disk_p2","SizeY (off track, diskp2)",100,0.,100.);
  meClSizeYNotOnTrack_diskp2->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYNotOnTrack_diskm1 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Disk_m1","SizeY (off track, diskm1)",100,0.,100.);
  meClSizeYNotOnTrack_diskm1->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeYNotOnTrack_diskm2 = dbe_->book1D("sizeY_" + clustersrc_.label() + "_Disk_m2","SizeY (off track, diskm2)",100,0.,100.);
  meClSizeYNotOnTrack_diskm2->setAxisTitle("Cluster size (in pixels)",1);
  

  //cluster global position
  //on track
  dbe_->setCurrentFolder("Pixel/Clusters/OnTrack");
  //bpix
  meClPosLayer1OnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_Layer_1","Clusters Layer1 (on track)",200,-30.,30.,128,-3.2,3.2);
  meClPosLayer1OnTrack->setAxisTitle("Global Z (cm)",1);
  meClPosLayer1OnTrack->setAxisTitle("Global #phi",2);
  meClPosLayer2OnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_Layer_2","Clusters Layer2 (on track)",200,-30.,30.,128,-3.2,3.2);
  meClPosLayer2OnTrack->setAxisTitle("Global Z (cm)",1);
  meClPosLayer2OnTrack->setAxisTitle("Global #phi",2);
  meClPosLayer3OnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_Layer_3","Clusters Layer3 (on track)",200,-30.,30.,128,-3.2,3.2);
  meClPosLayer3OnTrack->setAxisTitle("Global Z (cm)",1);
  meClPosLayer3OnTrack->setAxisTitle("Global #phi",2);
  //fpix
  meClPosDisk1pzOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_pz_Disk_1","Clusters +Z Disk1 (on track)",80,-20.,20.,80,-20.,20.);
  meClPosDisk1pzOnTrack->setAxisTitle("Global X (cm)",1);
  meClPosDisk1pzOnTrack->setAxisTitle("Global Y (cm)",2);
  meClPosDisk2pzOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_pz_Disk_2","Clusters +Z Disk2 (on track)",80,-20.,20.,80,-20.,20.);
  meClPosDisk2pzOnTrack->setAxisTitle("Global X (cm)",1);
  meClPosDisk2pzOnTrack->setAxisTitle("Global Y (cm)",2);
  meClPosDisk1mzOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_mz_Disk_1","Clusters -Z Disk1 (on track)",80,-20.,20.,80,-20.,20.);
  meClPosDisk1mzOnTrack->setAxisTitle("Global X (cm)",1);
  meClPosDisk1mzOnTrack->setAxisTitle("Global Y (cm)",2);
  meClPosDisk2mzOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_mz_Disk_2","Clusters -Z Disk2 (on track)",80,-20.,20.,80,-20.,20.);
  meClPosDisk2mzOnTrack->setAxisTitle("Global X (cm)",1);
  meClPosDisk2mzOnTrack->setAxisTitle("Global Y (cm)",2);

  meNClustersOnTrack_all = dbe_->book1D("nclusters_" + clustersrc_.label(),"Number of Clusters (on Track)",50,0.,50.);
  meNClustersOnTrack_all->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_bpix = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Barrel","Number of Clusters (on track, barrel)",50,0.,50.);
  meNClustersOnTrack_bpix->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_fpix = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Endcap","Number of Clusters (on track, endcap)",50,0.,50.);
  meNClustersOnTrack_fpix->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_layer1 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Layer_1","Number of Clusters (on track, layer1)",50,0.,50.);
  meNClustersOnTrack_layer1->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_layer2 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Layer_2","Number of Clusters (on track, layer2)",50,0.,50.);
  meNClustersOnTrack_layer2->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_layer3 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Layer_3","Number of Clusters (on track, layer3)",50,0.,50.);
  meNClustersOnTrack_layer3->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_diskp1 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Disk_p1","Number of Clusters (on track, diskp1)",50,0.,50.);
  meNClustersOnTrack_diskp1->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_diskp2 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Disk_p2","Number of Clusters (on track, diskp2)",50,0.,50.);
  meNClustersOnTrack_diskp2->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_diskm1 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Disk_m1","Number of Clusters (on track, diskm1)",50,0.,50.);
  meNClustersOnTrack_diskm1->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_diskm2 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Disk_m2","Number of Clusters (on track, diskm2)",50,0.,50.);
  meNClustersOnTrack_diskm2->setAxisTitle("Number of Clusters",1);

  //not on track
  dbe_->setCurrentFolder("Pixel/Clusters/OffTrack");
  //bpix
  meClPosLayer1NotOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_Layer_1","Clusters Layer1 (off track)",200,-30.,30.,128,-3.2,3.2);
  meClPosLayer1NotOnTrack->setAxisTitle("Global Z (cm)",1);
  meClPosLayer1NotOnTrack->setAxisTitle("Global #phi",2);
  meClPosLayer2NotOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_Layer_2","Clusters Layer2 (off track)",200,-30.,30.,128,-3.2,3.2);
  meClPosLayer2NotOnTrack->setAxisTitle("Global Z (cm)",1);
  meClPosLayer2NotOnTrack->setAxisTitle("Global #phi",2);
  meClPosLayer3NotOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_Layer_3","Clusters Layer3 (off track)",200,-30.,30.,128,-3.2,3.2);
  meClPosLayer3NotOnTrack->setAxisTitle("Global Z (cm)",1);
  meClPosLayer3NotOnTrack->setAxisTitle("Global #phi",2);
  //fpix
  meClPosDisk1pzNotOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_pz_Disk_1","Clusters +Z Disk1 (off track)",80,-20.,20.,80,-20.,20.);
  meClPosDisk1pzNotOnTrack->setAxisTitle("Global X (cm)",1);
  meClPosDisk1pzNotOnTrack->setAxisTitle("Global Y (cm)",2);
  meClPosDisk2pzNotOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_pz_Disk_2","Clusters +Z Disk2 (off track)",80,-20.,20.,80,-20.,20.);
  meClPosDisk2pzNotOnTrack->setAxisTitle("Global X (cm)",1);
  meClPosDisk2pzNotOnTrack->setAxisTitle("Global Y (cm)",2);
  meClPosDisk1mzNotOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_mz_Disk_1","Clusters -Z Disk1 (off track)",80,-20.,20.,80,-20.,20.);
  meClPosDisk1mzNotOnTrack->setAxisTitle("Global X (cm)",1);
  meClPosDisk1mzNotOnTrack->setAxisTitle("Global Y (cm)",2);
  meClPosDisk2mzNotOnTrack = dbe_->book2D("position_" + clustersrc_.label() + "_mz_Disk_2","Clusters -Z Disk2 (off track)",80,-20.,20.,80,-20.,20.);
  meClPosDisk2mzNotOnTrack->setAxisTitle("Global X (cm)",1);
  meClPosDisk2mzNotOnTrack->setAxisTitle("Global Y (cm)",2);

  meNClustersNotOnTrack_all = dbe_->book1D("nclusters_" + clustersrc_.label(),"Number of Clusters (off Track)",50,0.,50.);
  meNClustersNotOnTrack_all->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_bpix = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Barrel","Number of Clusters (off track, barrel)",50,0.,50.);
  meNClustersNotOnTrack_bpix->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_fpix = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Endcap","Number of Clusters (off track, endcap)",50,0.,50.);
  meNClustersNotOnTrack_fpix->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_layer1 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Layer_1","Number of Clusters (off track, layer1)",50,0.,50.);
  meNClustersNotOnTrack_layer1->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_layer2 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Layer_2","Number of Clusters (off track, layer2)",50,0.,50.);
  meNClustersNotOnTrack_layer2->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_layer3 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Layer_3","Number of Clusters (off track, layer3)",50,0.,50.);
  meNClustersNotOnTrack_layer3->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_diskp1 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Disk_p1","Number of Clusters (off track, diskp1)",50,0.,50.);
  meNClustersNotOnTrack_diskp1->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_diskp2 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Disk_p2","Number of Clusters (off track, diskp2)",50,0.,50.);
  meNClustersNotOnTrack_diskp2->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_diskm1 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Disk_m1","Number of Clusters (off track, diskm1)",50,0.,50.);
  meNClustersNotOnTrack_diskm1->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_diskm2 = dbe_->book1D("nclusters_" + clustersrc_.label() + "_Disk_m2","Number of Clusters (off track, diskm2)",50,0.,50.);
  meNClustersNotOnTrack_diskm2->setAxisTitle("Number of Clusters",1);

  //HitProbability
  //on track
  dbe_->setCurrentFolder("Pixel/Clusters/OnTrack");
  meHitProbability = dbe_->book1D("FractionLowProb","Fraction of hits with low probability;FractionLowProb;#HitsOnTrack",100,0.,1.);

  if (debug_) {
    // book summary residual histograms in a debugging folder - one (x,y) pair of histograms per subdetector 
    dbe_->setCurrentFolder("debugging"); 
    char hisID[80]; 
    for (int s=0; s<3; s++) {
      sprintf(hisID,"residual_x_subdet_%i",s); 
      meSubdetResidualX[s] = dbe_->book1D(hisID,"Pixel Hit-to-Track Residual in X",500,-5.,5.);
      
      sprintf(hisID,"residual_y_subdet_%i",s);
      meSubdetResidualY[s] = dbe_->book1D(hisID,"Pixel Hit-to-Track Residual in Y",500,-5.,5.);  
    }
  }
  
  firstRun = false;
  }
}


void SiPixelTrackResidualSource::endJob(void) {
  LogInfo("PixelDQM") << "SiPixelTrackResidualSource endJob()";

  // save the residual histograms to an output root file
  bool saveFile = pSet_.getUntrackedParameter<bool>("saveFile", true);
  if (saveFile) { 
    std::string outputFile = pSet_.getParameter<std::string>("outputFile");
    LogInfo("PixelDQM") << " - saving histograms to "<< outputFile.data();
    dbe_->save(outputFile);
  } 
  LogInfo("PixelDQM") << endl; // dbe_->showDirStructure();
}


void SiPixelTrackResidualSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  // retrieve TrackerGeometry again and MagneticField for use in transforming 
  // a TrackCandidate's P(ersistent)TrajectoryStateoOnDet (PTSoD) to a TrajectoryStateOnSurface (TSoS)
  ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  const TrackerGeometry* theTrackerGeometry = TG.product();
  ESHandle<MagneticField> MF;
  iSetup.get<IdealMagneticFieldRecord>().get(MF);
  const MagneticField* theMagneticField = MF.product();
  
  // retrieve TransientTrackingRecHitBuilder to build TTRHs with TrackCandidate's TrackingRecHits for refitting 
  std::string TTRHBuilder = pSet_.getParameter<std::string>("TTRHBuilder"); 
  ESHandle<TransientTrackingRecHitBuilder> TTRHB; 
  iSetup.get<TransientRecHitRecord>().get(TTRHBuilder, TTRHB);
  const TransientTrackingRecHitBuilder* theTTRHBuilder = TTRHB.product();
   
  // get a fitter to refit TrackCandidates, the same fitter as used in standard reconstruction 
  std::string Fitter = pSet_.getParameter<std::string>("Fitter");
  ESHandle<TrajectoryFitter> TF;
  iSetup.get<TrajectoryFitter::Record>().get(Fitter, TF);
  const TrajectoryFitter* theFitter = TF.product();

  // get TrackCandidateCollection in accordance with the fitter, i.e. rs-RS, ckf-KF... 
  std::string TrackCandidateLabel = pSet_.getParameter<std::string>("TrackCandidateLabel");
  std::string TrackCandidateProducer = pSet_.getParameter<std::string>("TrackCandidateProducer");  
  Handle<TrackCandidateCollection> trackCandidateCollection;
  iEvent.getByLabel(TrackCandidateProducer, TrackCandidateLabel, trackCandidateCollection);

  for (TrackCandidateCollection::const_iterator tc = trackCandidateCollection->begin(); 
       tc!=trackCandidateCollection->end(); ++tc) {
    PTrajectoryStateOnDet tcPTSoD = tc->trajectoryStateOnDet();
    TrajectoryStateOnSurface tcTSoS = trajectoryStateTransform::transientState(tcPTSoD, &(theTrackerGeometry->idToDet(tcPTSoD.detId())->surface()), 
						                 theMagneticField);
    const TrajectorySeed& tcSeed = tc->seed();

    const TrackCandidate::range& tcRecHits = tc->recHits();    
    if (debug_) cout << "track candidate has "<< int(tcRecHits.second - tcRecHits.first) <<" hits with ID "; 
    
    Trajectory::RecHitContainer tcTTRHs;
    for (TrackingRecHitCollection::const_iterator tcRecHit = tcRecHits.first; 
         tcRecHit!=tcRecHits.second; ++tcRecHit) { 
      if (debug_) cout << tcRecHit->geographicalId().rawId() <<" "; 
      
      tcTTRHs.push_back(theTTRHBuilder->build(&(*tcRecHit)));
    } 
    // note a TrackCandidate keeps only the PTSoD of the first hit as well as the seed and all the hits; 
    // to 99.9%-recover all the hit's TSoS's, refit with the seed, the hits and an initial TSoS from the PTSoD 
    // to get a Trajectory of all the hit's TrajectoryMeasurements (TMs) 
    std::vector<Trajectory> refitTrajectoryCollection = theFitter->fit(tcSeed, tcTTRHs, tcTSoS);	    
    if (debug_) cout << "refitTrajectoryCollection size is "<< refitTrajectoryCollection.size() << endl;

    if (refitTrajectoryCollection.size()>0) { // should be either 0 or 1 
      const Trajectory& refitTrajectory = refitTrajectoryCollection.front();

      // retrieve and loop over all the TMs 
      Trajectory::DataContainer refitTMs = refitTrajectory.measurements();								
      if (debug_) cout << "refitTrajectory has "<< refitTMs.size() <<" hits with ID "; 

      for (Trajectory::DataContainer::iterator refitTM = refitTMs.begin(); 
           refitTM!=refitTMs.end(); refitTM++) {  					
    	TransientTrackingRecHit::ConstRecHitPointer refitTTRH = refitTM->recHit();
        if (debug_) cout << refitTTRH->geographicalId().rawId() <<" "; 
	
	// only analyze the most elemental pixel hit's TMs to calculate residuals 
	const GeomDet* ttrhDet = refitTTRH->det(); 
	if (ttrhDet->components().empty() && (ttrhDet->subDetector()==GeomDetEnumerators::PixelBarrel ||				
    					      ttrhDet->subDetector()==GeomDetEnumerators::PixelEndcap)) {				

    	  // combine the forward and backward states without using the hit's information (hence unbiased by the hit); 
	  // the TM's updated state keeps the state combined and updated with the hit's info but we don't use the updated state at all 
	  TrajectoryStateOnSurface combinedTSoS = TrajectoryStateCombiner().combine(refitTM->forwardPredictedState(),        
    					        				    refitTM->backwardPredictedState());      
	  if (refitTTRH->isValid() && combinedTSoS.isValid()) { 
	    // calculate the distance between the hit location and the track-crossing point predicted by the combined state 
            const GeomDetUnit* GDU = static_cast<const GeomDetUnit*>(ttrhDet);
	    const Topology* theTopology = &(GDU->topology()); 									
    	    
	    MeasurementPoint hitPosition = theTopology->measurementPosition(refitTTRH->localPosition());				
    	    MeasurementPoint combinedTSoSPosition = theTopology->measurementPosition(combinedTSoS.localPosition());	
    	    
	    Measurement2DVector residual = hitPosition - combinedTSoSPosition;  			
	    if(debug_) std::cout << "fill residual " << residual.x() << " " << residual.y() << " \n";
    																
    	    // fill the residual histograms 
	    std::map<uint32_t, SiPixelTrackResidualModule*>::iterator pxd = theSiPixelStructure.find(refitTTRH->geographicalId().rawId());	
    	    if (pxd!=theSiPixelStructure.end()) (*pxd).second->fill(residual, reducedSet, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn);			
    																
    	    if (debug_) {
	      if (ttrhDet->subDetector()==GeomDetEnumerators::PixelBarrel) {			        		       
    	        meSubdetResidualX[0]->Fill(residual.x());					        		   
    	        meSubdetResidualY[0]->Fill(residual.y());					        		   
    	      } 										        		       
    	      else {										  
    	        meSubdetResidualX[PXFDetId(refitTTRH->geographicalId()).side()]->Fill(residual.x());  						    
    	        meSubdetResidualY[PXFDetId(refitTTRH->geographicalId()).side()]->Fill(residual.y());  						    
    	      } 
	    }															
    	  }
    	} 											
      } 
      if (debug_) cout << endl; 															
    }																
  } 


  //get trajectories
  edm::Handle<std::vector<Trajectory> > trajCollectionHandle;
  iEvent.getByLabel(tracksrc_,trajCollectionHandle);
  const std::vector<Trajectory> trajColl = *(trajCollectionHandle.product());
   
  //get tracks
  edm::Handle<std::vector<reco::Track> > trackCollectionHandle;
  iEvent.getByLabel(tracksrc_,trackCollectionHandle);
  const std::vector<reco::Track> trackColl = *(trackCollectionHandle.product());

  //get the map
  edm::Handle<TrajTrackAssociationCollection> match;
  iEvent.getByLabel(tracksrc_,match);
  const TrajTrackAssociationCollection ttac = *(match.product());

  // get clusters
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> >  clusterColl;
  iEvent.getByLabel( clustersrc_, clusterColl );
  const edmNew::DetSetVector<SiPixelCluster> clustColl = *(clusterColl.product());

  if(debug_){
    std::cout << "Trajectories\t : " << trajColl.size() << std::endl;
    std::cout << "recoTracks  \t : " << trackColl.size() << std::endl;
    std::cout << "Map entries \t : " << ttac.size() << std::endl;
  }

  std::set<SiPixelCluster> clusterSet;
  TrajectoryStateCombiner tsoscomb;
  int tracks=0, pixeltracks=0, bpixtracks=0, fpixtracks=0; 
  int trackclusters=0, barreltrackclusters=0, endcaptrackclusters=0;
  int otherclusters=0, barrelotherclusters=0, endcapotherclusters=0;

  //Loop over map entries
  for(TrajTrackAssociationCollection::const_iterator it =  ttac.begin();it !=  ttac.end(); ++it){
    const edm::Ref<std::vector<Trajectory> > traj_iterator = it->key;  
    // Trajectory Map, extract Trajectory for this track
    reco::TrackRef trackref = it->val;
    tracks++;

    bool isBpixtrack = false, isFpixtrack = false, crossesPixVol=false;
    
    //find out whether track crosses pixel fiducial volume (for cosmic tracks)
    
    double d0 = (*trackref).d0(), dz = (*trackref).dz(); 
    
    if(abs(d0)<15 && abs(dz)<50) crossesPixVol = true;

    std::vector<TrajectoryMeasurement> tmeasColl =traj_iterator->measurements();
    std::vector<TrajectoryMeasurement>::const_iterator tmeasIt;
    //loop on measurements to find out whether there are bpix and/or fpix hits
    for(tmeasIt = tmeasColl.begin();tmeasIt!=tmeasColl.end();tmeasIt++){
      if(! tmeasIt->updatedState().isValid()) continue; 
      TransientTrackingRecHit::ConstRecHitPointer testhit = tmeasIt->recHit();
      if(! testhit->isValid() || testhit->geographicalId().det() != DetId::Tracker) continue; 
      uint testSubDetID = (testhit->geographicalId().subdetId()); 
      if(testSubDetID==PixelSubdetector::PixelBarrel) isBpixtrack = true;
      if(testSubDetID==PixelSubdetector::PixelEndcap) isFpixtrack = true;
    }//end loop on measurements
    if(isBpixtrack) {
      bpixtracks++; 
      if(debug_) std::cout << "bpixtrack\n";
    }
    if(isFpixtrack) {
      fpixtracks++; 
      if(debug_) std::cout << "fpixtrack\n";
    }
    if(isBpixtrack || isFpixtrack){
      pixeltracks++;
      
      if(crossesPixVol) meNofTracksInPixVol_->Fill(0,1);

      std::vector<TrajectoryMeasurement> tmeasColl = traj_iterator->measurements();
      for(std::vector<TrajectoryMeasurement>::const_iterator tmeasIt = tmeasColl.begin(); tmeasIt!=tmeasColl.end(); tmeasIt++){   
	if(! tmeasIt->updatedState().isValid()) continue; 
	
	TrajectoryStateOnSurface tsos = tsoscomb( tmeasIt->forwardPredictedState(), tmeasIt->backwardPredictedState() );
	TransientTrackingRecHit::ConstRecHitPointer hit = tmeasIt->recHit();
	if(! hit->isValid() || hit->geographicalId().det() != DetId::Tracker ) {
	  continue; 
	} else {
	  
// 	  //residual
      	  const DetId & hit_detId = hit->geographicalId();
	  //uint IntRawDetID = (hit_detId.rawId());
	  uint IntSubDetID = (hit_detId.subdetId());
	  
 	  if(IntSubDetID == 0 ) continue; // don't look at SiStrip hits!	    

	  // get the enclosed persistent hit
	  const TrackingRecHit *persistentHit = hit->hit();
	  // check if it's not null, and if it's a valid pixel hit
	  if ((persistentHit != 0) && (typeid(*persistentHit) == typeid(SiPixelRecHit))) {
	    // tell the C++ compiler that the hit is a pixel hit
	    const SiPixelRecHit* pixhit = static_cast<const SiPixelRecHit*>( hit->hit() );
	    //Hit probability:
	    float hit_prob = -1.;
	    if(pixhit->hasFilledProb()){
	      hit_prob = pixhit->clusterProbability(0);
	      //std::cout<<"HITPROB= "<<hit_prob<<std::endl;
	      if(hit_prob<pow(10.,-15.)) NLowProb++;
	      NTotal++;
	      if(NTotal>0) meHitProbability->Fill(float(NLowProb/NTotal));
	    }
	    
	    // get the edm::Ref to the cluster
 	    edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = (*pixhit).cluster();
	    //  check if the ref is not null
	    if (clust.isNonnull()) {

	      //define tracker and pixel geometry and topology
	      const TrackerGeometry& theTracker(*theTrackerGeometry);
	      const PixelGeomDetUnit* theGeomDet = static_cast<const PixelGeomDetUnit*> (theTracker.idToDet(hit_detId) );
	      //test if PixelGeomDetUnit exists
	      if(theGeomDet == 0) {
		if(debug_) std::cout << "NO THEGEOMDET\n";
		continue;
	      }

	      const PixelTopology * topol = &(theGeomDet->specificTopology());
	      //fill histograms for clusters on tracks
	      //correct SiPixelTrackResidualModule
	      std::map<uint32_t, SiPixelTrackResidualModule*>::iterator pxd = theSiPixelStructure.find((*hit).geographicalId().rawId());

	      //CHARGE CORRECTION (for track impact angle)
	      // calculate alpha and beta from cluster position
	      LocalTrajectoryParameters ltp = tsos.localParameters();
	      LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
	      
	      float clust_alpha = atan2(localDir.z(), localDir.x());
	      float clust_beta = atan2(localDir.z(), localDir.y());
	      double corrCharge = clust->charge() * sqrt( 1.0 / ( 1.0/pow( tan(clust_alpha), 2 ) + 
								  1.0/pow( tan(clust_beta ), 2 ) + 
								  1.0 )
							  )/1000.;

	      if (pxd!=theSiPixelStructure.end()) (*pxd).second->fill((*clust), true, corrCharge, reducedSet, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn); 	


	      trackclusters++;
	      //CORR CHARGE
	      meClChargeOnTrack_all->Fill(corrCharge);
	      meClSizeOnTrack_all->Fill((*clust).size());
	      meClSizeXOnTrack_all->Fill((*clust).sizeX());
	      meClSizeYOnTrack_all->Fill((*clust).sizeY());
	      clusterSet.insert(*clust);

	      //find cluster global position (rphi, z)
	      // get cluster center of gravity (of charge)
	      float xcenter = clust->x();
	      float ycenter = clust->y();
	      // get the cluster position in local coordinates (cm) 
	      LocalPoint clustlp = topol->localPosition( MeasurementPoint(xcenter, ycenter) );
	      // get the cluster position in global coordinates (cm)
	      GlobalPoint clustgp = theGeomDet->surface().toGlobal( clustlp );

	      //find location of hit (barrel or endcap, same for cluster)
	      bool barrel = DetId((*hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
	      bool endcap = DetId((*hit).geographicalId()).subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap);
	      if(barrel) {
		barreltrackclusters++;
		//CORR CHARGE
		meClChargeOnTrack_bpix->Fill(corrCharge);
		meClSizeOnTrack_bpix->Fill((*clust).size());
		meClSizeXOnTrack_bpix->Fill((*clust).sizeX());
		meClSizeYOnTrack_bpix->Fill((*clust).sizeY());
		uint32_t DBlayer = PixelBarrelName(DetId((*hit).geographicalId())).layerName();
		float phi = clustgp.phi(); 
		float z = clustgp.z();
		switch(DBlayer){
		case 1: {
		  meClPosLayer1OnTrack->Fill(z,phi); 
		  meClChargeOnTrack_layer1->Fill(corrCharge);
		  meClSizeOnTrack_layer1->Fill((*clust).size());
		  meClSizeXOnTrack_layer1->Fill((*clust).sizeX());
		  meClSizeYOnTrack_layer1->Fill((*clust).sizeY());
		  break;
		}
		case 2: {
		  meClPosLayer2OnTrack->Fill(z,phi); 
		  meClChargeOnTrack_layer2->Fill(corrCharge);
		  meClSizeOnTrack_layer2->Fill((*clust).size());
		  meClSizeXOnTrack_layer2->Fill((*clust).sizeX());
		  meClSizeYOnTrack_layer2->Fill((*clust).sizeY());
		  break;
		}
		case 3: {
		  meClPosLayer3OnTrack->Fill(z,phi); 
		  meClChargeOnTrack_layer3->Fill(corrCharge);
		  meClSizeOnTrack_layer3->Fill((*clust).size());
		  meClSizeXOnTrack_layer3->Fill((*clust).sizeX());
		  meClSizeYOnTrack_layer3->Fill((*clust).sizeY());
		  break;
		}
		
		}
		
	      }
	      if(endcap) {
		endcaptrackclusters++;
		//CORR CHARGE
		meClChargeOnTrack_fpix->Fill(corrCharge);
		meClSizeOnTrack_fpix->Fill((*clust).size());
		meClSizeXOnTrack_fpix->Fill((*clust).sizeX());
		meClSizeYOnTrack_fpix->Fill((*clust).sizeY());
		uint32_t DBdisk = PixelEndcapName(DetId((*hit).geographicalId())).diskName();
		float x = clustgp.x(); 
		float y = clustgp.y(); 
		float z = clustgp.z();
		if(z>0){
		  if(DBdisk==1) {
		    meClPosDisk1pzOnTrack->Fill(x,y);
		    meClChargeOnTrack_diskp1->Fill(corrCharge);
		    meClSizeOnTrack_diskp1->Fill((*clust).size());
		    meClSizeXOnTrack_diskp1->Fill((*clust).sizeX());
		    meClSizeYOnTrack_diskp1->Fill((*clust).sizeY());
		  }
		  if(DBdisk==2) {
		    meClPosDisk2pzOnTrack->Fill(x,y); 
		    meClChargeOnTrack_diskp2->Fill(corrCharge);
		    meClSizeOnTrack_diskp2->Fill((*clust).size());
		    meClSizeXOnTrack_diskp2->Fill((*clust).sizeX());
		    meClSizeYOnTrack_diskp2->Fill((*clust).sizeY());
		  }
		}
		else{
		  if(DBdisk==1) {
		    meClPosDisk1mzOnTrack->Fill(x,y);
		    meClChargeOnTrack_diskm1->Fill(corrCharge);
		    meClSizeOnTrack_diskm1->Fill((*clust).size());
		    meClSizeXOnTrack_diskm1->Fill((*clust).sizeX());
		    meClSizeYOnTrack_diskm1->Fill((*clust).sizeY());
		  }
		  if(DBdisk==2) {
		    meClPosDisk2mzOnTrack->Fill(x,y); 
		    meClChargeOnTrack_diskm2->Fill(corrCharge);
		    meClSizeOnTrack_diskm2->Fill((*clust).size());
		    meClSizeXOnTrack_diskm2->Fill((*clust).sizeX());
		    meClSizeYOnTrack_diskm2->Fill((*clust).sizeY());
		  }
		} 
	      }
	      
	    }//end if (cluster exists)

	  }//end if (persistent hit exists and is pixel hit)
	  
	}//end of else 
	
	
      }//end for (all traj measurements of pixeltrack)
    }//end if (is pixeltrack)
    else {
      if(debug_) std::cout << "no pixeltrack:\n";
      if(crossesPixVol) meNofTracksInPixVol_->Fill(1,1);
    }
    
  }//end loop on map entries

  //find clusters that are NOT on track
  //edmNew::DetSet<SiPixelCluster>::const_iterator  di;
  if(debug_) std::cout << "clusters not on track: (size " << clustColl.size() << ") ";

  for(TrackerGeometry::DetContainer::const_iterator it = TG->dets().begin(); it != TG->dets().end(); it++){
    //if(dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
    DetId detId = (*it)->geographicalId();
    if(detId>=302055684 && detId<=352477708){ // make sure it's a Pixel module WITHOUT using dynamic_cast!  
      int nofclOnTrack = 0, nofclOffTrack=0; 
      uint32_t DBlayer=10, DBdisk=10; 
      float z=0.; 
      //set layer/disk
      if(DetId(detId).subdetId() == 1) { // Barrel module
	DBlayer = PixelBarrelName(DetId(detId)).layerName();
      }
      if(DetId(detId).subdetId() == 2){ // Endcap module
	DBdisk = PixelEndcapName(DetId(detId )).diskName();
      }
      edmNew::DetSetVector<SiPixelCluster>::const_iterator isearch = clustColl.find(detId);
      if( isearch != clustColl.end() ) {  // Not an empty iterator
	edmNew::DetSet<SiPixelCluster>::const_iterator  di;
	for(di=isearch->begin(); di!=isearch->end(); di++){
	  unsigned int temp = clusterSet.size();
	  clusterSet.insert(*di);
	  //check if cluster is off track
	  if(clusterSet.size()>temp) {
	    otherclusters++;
	    nofclOffTrack++; 
	    //fill histograms for clusters off tracks
	    //correct SiPixelTrackResidualModule
	    std::map<uint32_t, SiPixelTrackResidualModule*>::iterator pxd = theSiPixelStructure.find((*it)->geographicalId().rawId());

	    if (pxd!=theSiPixelStructure.end()) (*pxd).second->fill((*di), false, -1., reducedSet, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn); 
	    


	    meClSizeNotOnTrack_all->Fill((*di).size());
	    meClSizeXNotOnTrack_all->Fill((*di).sizeX());
	    meClSizeYNotOnTrack_all->Fill((*di).sizeY());
	    meClChargeNotOnTrack_all->Fill((*di).charge()/1000);

	    /////////////////////////////////////////////////
	    //find cluster global position (rphi, z) get cluster
	    //define tracker and pixel geometry and topology
	    const TrackerGeometry& theTracker(*theTrackerGeometry);
	    const PixelGeomDetUnit* theGeomDet = static_cast<const PixelGeomDetUnit*> (theTracker.idToDet(detId) );
	    //test if PixelGeomDetUnit exists
	    if(theGeomDet == 0) {
	      if(debug_) std::cout << "NO THEGEOMDET\n";
	      continue;
	    }
	    const PixelTopology * topol = &(theGeomDet->specificTopology());
	   
	    //center of gravity (of charge)
	    float xcenter = di->x();
	    float ycenter = di->y();
	    // get the cluster position in local coordinates (cm) 
	    LocalPoint clustlp = topol->localPosition( MeasurementPoint(xcenter, ycenter) );
	    // get the cluster position in global coordinates (cm)
	    GlobalPoint clustgp = theGeomDet->surface().toGlobal( clustlp );

	    /////////////////////////////////////////////////

	    //barrel
	    if(DetId(detId).subdetId() == 1) {
	      meClSizeNotOnTrack_bpix->Fill((*di).size());
	      meClSizeXNotOnTrack_bpix->Fill((*di).sizeX());
	      meClSizeYNotOnTrack_bpix->Fill((*di).sizeY());
	      meClChargeNotOnTrack_bpix->Fill((*di).charge()/1000);
	      barrelotherclusters++;
	      //DBlayer = PixelBarrelName(DetId(detId)).layerName();
	      float phi = clustgp.phi(); 
	      //float r = clustgp.perp();
	      z = clustgp.z();
	      switch(DBlayer){
	      case 1: {
		meClPosLayer1NotOnTrack->Fill(z,phi); 
		meClSizeNotOnTrack_layer1->Fill((*di).size());
		meClSizeXNotOnTrack_layer1->Fill((*di).sizeX());
		meClSizeYNotOnTrack_layer1->Fill((*di).sizeY());
		meClChargeNotOnTrack_layer1->Fill((*di).charge()/1000);
		break;
	      }
	      case 2: {
		meClPosLayer2NotOnTrack->Fill(z,phi);
		meClSizeNotOnTrack_layer2->Fill((*di).size());
		meClSizeXNotOnTrack_layer2->Fill((*di).sizeX());
		meClSizeYNotOnTrack_layer2->Fill((*di).sizeY());
		meClChargeNotOnTrack_layer2->Fill((*di).charge()/1000);
		break;
	      }
	      case 3: {
		meClPosLayer3NotOnTrack->Fill(z,phi); 
		meClSizeNotOnTrack_layer3->Fill((*di).size());
		meClSizeXNotOnTrack_layer3->Fill((*di).sizeX());
		meClSizeYNotOnTrack_layer3->Fill((*di).sizeY());
		meClChargeNotOnTrack_layer3->Fill((*di).charge()/1000);
		break;
	      }
		
	      }
	    }
	    //endcap
	    if(DetId(detId).subdetId() == 2) {
	      meClSizeNotOnTrack_fpix->Fill((*di).size());
	      meClSizeXNotOnTrack_fpix->Fill((*di).sizeX());
	      meClSizeYNotOnTrack_fpix->Fill((*di).sizeY());
	      meClChargeNotOnTrack_fpix->Fill((*di).charge()/1000);
	      endcapotherclusters++;
	      //DBdisk = PixelEndcapName(DetId(detId )).diskName();
	      float x = clustgp.x(); 
	      float y = clustgp.y(); 
	      z = clustgp.z();
	      if(z>0){
		if(DBdisk==1) {
		  meClPosDisk1pzNotOnTrack->Fill(x,y);
		  meClSizeNotOnTrack_diskp1->Fill((*di).size());
		  meClSizeXNotOnTrack_diskp1->Fill((*di).sizeX());
		  meClSizeYNotOnTrack_diskp1->Fill((*di).sizeY());
		  meClChargeNotOnTrack_diskp1->Fill((*di).charge()/1000);
		}
		if(DBdisk==2) {
		  meClPosDisk2pzNotOnTrack->Fill(x,y); 
		  meClSizeNotOnTrack_diskp2->Fill((*di).size());
		  meClSizeXNotOnTrack_diskp2->Fill((*di).sizeX());
		  meClSizeYNotOnTrack_diskp2->Fill((*di).sizeY());
		  meClChargeNotOnTrack_diskp2->Fill((*di).charge()/1000);
		}
	      }
	      else{
		if(DBdisk==1) {
		  meClPosDisk1mzNotOnTrack->Fill(x,y);
		  meClSizeNotOnTrack_diskm1->Fill((*di).size());
		  meClSizeXNotOnTrack_diskm1->Fill((*di).sizeX());
		  meClSizeYNotOnTrack_diskm1->Fill((*di).sizeY());
		  meClChargeNotOnTrack_diskm1->Fill((*di).charge()/1000);
		}
		if(DBdisk==2) {
		  meClPosDisk2mzNotOnTrack->Fill(x,y); 
		  meClSizeNotOnTrack_diskm2->Fill((*di).size());
		  meClSizeXNotOnTrack_diskm2->Fill((*di).sizeX());
		  meClSizeYNotOnTrack_diskm2->Fill((*di).sizeY());
		  meClChargeNotOnTrack_diskm2->Fill((*di).charge()/1000);
		}
	      } 

	    }
	  }// end "if cluster off track"
	  else {
	    nofclOnTrack++; 
	    if(z == 0 && DBdisk != 10){
	      //find cluster global position (rphi, z) get cluster
	      //define tracker and pixel geometry and topology
	      const TrackerGeometry& theTracker(*theTrackerGeometry);
	      const PixelGeomDetUnit* theGeomDet = static_cast<const PixelGeomDetUnit*> (theTracker.idToDet(detId) );
	      //test if PixelGeomDetUnit exists
	      if(theGeomDet == 0) {
		if(debug_) std::cout << "NO THEGEOMDET\n";
		continue;
	      }
	      const PixelTopology * topol = &(theGeomDet->specificTopology());
	      //center of gravity (of charge)
	      float xcenter = di->x();
	      float ycenter = di->y();
	      // get the cluster position in local coordinates (cm) 
	      LocalPoint clustlp = topol->localPosition( MeasurementPoint(xcenter, ycenter) );
	      // get the cluster position in global coordinates (cm)
	      GlobalPoint clustgp = theGeomDet->surface().toGlobal( clustlp );  
	      z = clustgp.z();
	    }
	  }
	}
      }
      //++ fill the number of clusters on a module
      std::map<uint32_t, SiPixelTrackResidualModule*>::iterator pxd = theSiPixelStructure.find((*it)->geographicalId().rawId());
      if (pxd!=theSiPixelStructure.end()) (*pxd).second->nfill(nofclOnTrack, nofclOffTrack, reducedSet, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn); 
      if(nofclOnTrack!=0) meNClustersOnTrack_all->Fill(nofclOnTrack); 
      if(nofclOffTrack!=0) meNClustersNotOnTrack_all->Fill(nofclOffTrack); 
      //barrel
      if(DetId(detId).subdetId() == 1){
	if(nofclOnTrack!=0) meNClustersOnTrack_bpix->Fill(nofclOnTrack); 
	if(nofclOffTrack!=0) meNClustersNotOnTrack_bpix->Fill(nofclOffTrack); 
	//DBlayer = PixelBarrelName(DetId(detId)).layerName();
	switch(DBlayer){
	case 1: { 
	  if(nofclOnTrack!=0) meNClustersOnTrack_layer1->Fill(nofclOnTrack);
	  if(nofclOffTrack!=0) meNClustersNotOnTrack_layer1->Fill(nofclOffTrack); break;
	}
	case 2: {
	  if(nofclOnTrack!=0) meNClustersOnTrack_layer2->Fill(nofclOnTrack);
	  if(nofclOffTrack!=0) meNClustersNotOnTrack_layer2->Fill(nofclOffTrack); break; 
	}
	case 3: {
	  if(nofclOnTrack!=0) meNClustersOnTrack_layer3->Fill(nofclOnTrack); 
	  if(nofclOffTrack!=0) meNClustersNotOnTrack_layer3->Fill(nofclOffTrack); break; 
	}
	}
      }//end barrel
      //endcap
      if(DetId(detId).subdetId() == 2) {
	//DBdisk = PixelEndcapName(DetId(detId )).diskName();
	//z = clustgp.z();
	if(nofclOnTrack!=0) meNClustersOnTrack_fpix->Fill(nofclOnTrack); 
	if(nofclOffTrack!=0) meNClustersNotOnTrack_fpix->Fill(nofclOffTrack);
	if(z>0){
	  if(DBdisk==1) {
	    if(nofclOnTrack!=0) meNClustersOnTrack_diskp1->Fill(nofclOnTrack); 
	    if(nofclOffTrack!=0) meNClustersNotOnTrack_diskp1->Fill(nofclOffTrack);
	  }
	  if(DBdisk==2) {
	    if(nofclOnTrack!=0) meNClustersOnTrack_diskp2->Fill(nofclOnTrack); 
	    if(nofclOffTrack!=0) meNClustersNotOnTrack_diskp2->Fill(nofclOffTrack);
	  }
	}
	if(z<0){
	  if(DBdisk==1) {
	    if(nofclOnTrack!=0) meNClustersOnTrack_diskm1->Fill(nofclOnTrack); 
	    if(nofclOffTrack!=0) meNClustersNotOnTrack_diskm1->Fill(nofclOffTrack);
	  }
	  if(DBdisk==2) {
	    if(nofclOnTrack!=0) meNClustersOnTrack_diskm2->Fill(nofclOnTrack); 
	    if(nofclOffTrack!=0) meNClustersNotOnTrack_diskm2->Fill(nofclOffTrack);
	  }
	}
      }

    }//end if it's a Pixel module
  }//end for loop over tracker detector geometry modules


  if(trackclusters>0) (meNofClustersOnTrack_)->Fill(0,trackclusters);
  if(barreltrackclusters>0)(meNofClustersOnTrack_)->Fill(1,barreltrackclusters);
  if(endcaptrackclusters>0)(meNofClustersOnTrack_)->Fill(2,endcaptrackclusters);
  if(otherclusters>0)(meNofClustersNotOnTrack_)->Fill(0,otherclusters);
  if(barrelotherclusters>0)(meNofClustersNotOnTrack_)->Fill(1,barrelotherclusters);
  if(endcapotherclusters>0)(meNofClustersNotOnTrack_)->Fill(2,endcapotherclusters);
  if(tracks>0)(meNofTracks_)->Fill(0,tracks);
  if(pixeltracks>0)(meNofTracks_)->Fill(1,pixeltracks);
  if(bpixtracks>0)(meNofTracks_)->Fill(2,bpixtracks);
  if(fpixtracks>0)(meNofTracks_)->Fill(3,fpixtracks);
}


DEFINE_FWK_MODULE(SiPixelTrackResidualSource); // define this as a plug-in 
