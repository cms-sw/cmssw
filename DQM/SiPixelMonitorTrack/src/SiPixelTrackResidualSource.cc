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


#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
//#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameWrapper.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
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

//Claudia new libraries
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

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
  diskOn( pSet.getUntrackedParameter<bool>("diskOn",false) ),
  isUpgrade( pSet.getUntrackedParameter<bool>("isUpgrade",false) ),
  noOfLayers(0),
  noOfDisks(0)
{ 
   pSet_ = pSet; 
   debug_ = pSet_.getUntrackedParameter<bool>("debug", false); 
   src_ = pSet_.getParameter<edm::InputTag>("src"); 
   clustersrc_ = pSet_.getParameter<edm::InputTag>("clustersrc");
   tracksrc_ = pSet_.getParameter<edm::InputTag>("trajectoryInput");
   ttrhbuilder_ = pSet_.getParameter<std::string>("TTRHBuilder");
   ptminres_= pSet.getUntrackedParameter<double>("PtMinRes",4.0) ;
   beamSpotToken_ = consumes<reco::BeamSpot>(std::string("offlineBeamSpot"));
   vtxsrc_=pSet_.getUntrackedParameter<std::string>("vtxsrc",  "offlinePrimaryVertices");
   offlinePrimaryVerticesToken_ =  consumes<reco::VertexCollection>(vtxsrc_);// consumes<reco::VertexCollection>(std::string("hiSelectedVertex"));     //"offlinePrimaryVertices"));
   generalTracksToken_ = consumes<reco::TrackCollection>(pSet_.getParameter<edm::InputTag>("tracksrc"));
   tracksrcToken_ = consumes<std::vector<Trajectory> >(pSet_.getParameter<edm::InputTag>("trajectoryInput"));
   trackToken_ = consumes<std::vector<reco::Track> >(pSet_.getParameter<edm::InputTag>("trajectoryInput"));
   trackAssociationToken_ = consumes<TrajTrackAssociationCollection>(pSet_.getParameter<edm::InputTag>("trajectoryInput"));
   clustersrcToken_ = consumes<edmNew::DetSetVector<SiPixelCluster> >(pSet_.getParameter<edm::InputTag>("clustersrc"));
   digisrc_ = pSet_.getParameter<edm::InputTag>("digisrc");
   digisrcToken_ = consumes<edm::DetSetVector<PixelDigi> >(pSet_.getParameter<edm::InputTag>("digisrc"));

  LogInfo("PixelDQM") << "SiPixelTrackResidualSource constructor" << endl;
  LogInfo ("PixelDQM") << "Mod/Lad/Lay/Phi " << modOn << "/" << ladOn << "/" 
            << layOn << "/" << phiOn << std::endl;
  LogInfo ("PixelDQM") << "Blade/Disk/Ring" << bladeOn << "/" << diskOn << "/" 
            << ringOn << std::endl;

  topFolderName_ = pSet_.getParameter<std::string>("TopFolderName");

  firstRun = true;
  NTotal=0;
  NLowProb=0;
}


SiPixelTrackResidualSource::~SiPixelTrackResidualSource() {
  LogInfo("PixelDQM") << "SiPixelTrackResidualSource destructor" << endl;

  std::map<uint32_t,SiPixelTrackResidualModule*>::iterator struct_iter;
  for (struct_iter = theSiPixelStructure.begin() ; struct_iter != theSiPixelStructure.end() ; struct_iter++){
    delete struct_iter->second;
    struct_iter->second = 0;
  }
}


void SiPixelTrackResidualSource::dqmBeginRun(const edm::Run& r, edm::EventSetup const& iSetup) {
  LogInfo("PixelDQM") << "SiPixelTrackResidualSource beginRun()" << endl;
  // retrieve TrackerGeometry for pixel dets
  edm::ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  if (debug_) LogVerbatim("PixelDQM") << "TrackerGeometry "<< &(*TG) <<" size is "<< TG->dets().size() << endl;
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology *pTT = tTopoHandle.product();

  // build theSiPixelStructure with the pixel barrel and endcap dets from TrackerGeometry
  for (TrackerGeometry::DetContainer::const_iterator pxb = TG->detsPXB().begin();  
       pxb!=TG->detsPXB().end(); pxb++) {
    if (dynamic_cast<PixelGeomDetUnit const *>((*pxb))!=0) {
      SiPixelTrackResidualModule* module = new SiPixelTrackResidualModule((*pxb)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelTrackResidualModule*>((*pxb)->geographicalId().rawId(), module));
      //int DBlayer = PixelBarrelNameWrapper(pSet_, DetId((*pxb)->geographicalId())).layerName();
      int DBlayer = PixelBarrelName(DetId((*pxb)->geographicalId()),pTT,isUpgrade).layerName();
      if (noOfLayers < DBlayer) noOfLayers = DBlayer;
    }
  }
  for (TrackerGeometry::DetContainer::const_iterator pxf = TG->detsPXF().begin(); 
       pxf!=TG->detsPXF().end(); pxf++) {
    if (dynamic_cast<PixelGeomDetUnit const *>((*pxf))!=0) {
      SiPixelTrackResidualModule* module = new SiPixelTrackResidualModule((*pxf)->geographicalId().rawId());
      theSiPixelStructure.insert(pair<uint32_t, SiPixelTrackResidualModule*>((*pxf)->geographicalId().rawId(), module));
      int DBdisk;
      DBdisk = PixelEndcapName(DetId((*pxf)->geographicalId()),pTT,isUpgrade).diskName();
      if (noOfDisks < DBdisk) noOfDisks = DBdisk;
    }
  }
  LogInfo("PixelDQM") << "SiPixelStructure size is " << theSiPixelStructure.size() << endl;
}

void SiPixelTrackResidualSource::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const & iSetup){
 
  // book residual histograms in theSiPixelFolder - one (x,y) pair of histograms per det
  SiPixelFolderOrganizer theSiPixelFolder(false);
  std::stringstream nameX, titleX, nameY, titleY;

  if(ladOn){
    iBooker.setCurrentFolder(topFolderName_+"/Barrel");
    for (int i = 1; i <= noOfLayers; i++){
      nameX.str(std::string()); nameX <<"siPixelTrackResidualsX_SummedLayer_" << i;
      titleX.str(std::string()); titleX <<"Layer"<< i << "Hit-to-Track Residual in r-phi";
      meResidualXSummedLay.push_back(iBooker.book1D(nameX.str(),titleX.str(),100,-150,150));
      meResidualXSummedLay.at(i-1)->setAxisTitle("hit-to-track residual in r-phi (um)",1);
      nameY.str(std::string()); nameY <<"siPixelTrackResidualsY_SummedLayer_" << i;
      titleY.str(std::string()); titleY <<"Layer"<< i << "Hit-to-Track Residual in Z";
      meResidualYSummedLay.push_back(iBooker.book1D(nameY.str(),titleY.str(),100,-300,300));
      meResidualYSummedLay.at(i-1)->setAxisTitle("hit-to-track residual in z (um)",1);
    }
  }

  for (std::map<uint32_t, SiPixelTrackResidualModule*>::iterator pxd = theSiPixelStructure.begin(); 
       pxd!=theSiPixelStructure.end(); pxd++){
   
    if(modOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,0,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,reducedSet,0,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource Folder Creation Failed! "; 
    }
    if(ladOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,1,isUpgrade)) {
	
	(*pxd).second->book(pSet_,iSetup,iBooker,reducedSet,1,isUpgrade);
      }
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource ladder Folder Creation Failed! "; 
    }
    if(layOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,2,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,reducedSet,2,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource layer Folder Creation Failed! "; 
    }
    if(phiOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,3,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,reducedSet,3,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource phi Folder Creation Failed! "; 
    }
    if(bladeOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,4,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,reducedSet,4,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource Blade Folder Creation Failed! "; 
    }
    if(diskOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,5,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,reducedSet,5,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource Disk Folder Creation Failed! "; 
    }
    if(ringOn){
      if (theSiPixelFolder.setModuleFolder(iBooker,(*pxd).first,6,isUpgrade)) (*pxd).second->book(pSet_,iSetup,iBooker,reducedSet,6,isUpgrade);
      else throw cms::Exception("LogicError") << "SiPixelTrackResidualSource Ring Folder Creation Failed! "; 
    }
  }

//   edm::InputTag tracksrc = pSet_.getParameter<edm::InputTag>("trajectoryInput");
//   edm::InputTag clustersrc = pSet_.getParameter<edm::InputTag>("clustersrc");

  //number of tracks
  iBooker.setCurrentFolder(topFolderName_+"/Tracks");
  meNofTracks_ = iBooker.book1D("ntracks_" + tracksrc_.label(),"Number of Tracks",4,0,4);
  meNofTracks_->setAxisTitle("Number of Tracks",1);
  meNofTracks_->setBinLabel(1,"All");
  meNofTracks_->setBinLabel(2,"Pixel");
  meNofTracks_->setBinLabel(3,"BPix");
  meNofTracks_->setBinLabel(4,"FPix");

  //number of tracks in pixel fiducial volume
  iBooker.setCurrentFolder(topFolderName_+"/Tracks");
  meNofTracksInPixVol_ = iBooker.book1D("ntracksInPixVol_" + tracksrc_.label(),"Number of Tracks crossing Pixel fiducial Volume",2,0,2);
  meNofTracksInPixVol_->setAxisTitle("Number of Tracks",1);
  meNofTracksInPixVol_->setBinLabel(1,"With Hits");
  meNofTracksInPixVol_->setBinLabel(2,"Without Hits");

  //number of clusters (associated to track / not associated)
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OnTrack");
  meNofClustersOnTrack_ = iBooker.book1D("nclusters_" + clustersrc_.label() + "_tot","Number of Clusters (on track)",3,0,3);
  meNofClustersOnTrack_->setAxisTitle("Number of Clusters on Track",1);
  meNofClustersOnTrack_->setBinLabel(1,"All");
  meNofClustersOnTrack_->setBinLabel(2,"BPix");
  meNofClustersOnTrack_->setBinLabel(3,"FPix");
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OffTrack");
  meNofClustersNotOnTrack_ = iBooker.book1D("nclusters_" + clustersrc_.label() + "_tot","Number of Clusters (off track)",3,0,3);
  meNofClustersNotOnTrack_->setAxisTitle("Number of Clusters off Track",1);
  meNofClustersNotOnTrack_->setBinLabel(1,"All");
  meNofClustersNotOnTrack_->setBinLabel(2,"BPix");
  meNofClustersNotOnTrack_->setBinLabel(3,"FPix");

  //cluster charge and size
  //charge
  //on track
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OnTrack");
  std::stringstream ss1, ss2;
  meClChargeOnTrack_all = iBooker.book1D("charge_" + clustersrc_.label(),"Charge (on track)",500,0.,500.);
  meClChargeOnTrack_all->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_bpix = iBooker.book1D("charge_" + clustersrc_.label() + "_Barrel","Charge (on track, barrel)",500,0.,500.);
  meClChargeOnTrack_bpix->setAxisTitle("Charge size (in ke)",1);
  meClChargeOnTrack_fpix = iBooker.book1D("charge_" + clustersrc_.label() + "_Endcap","Charge (on track, endcap)",500,0.,500.);
  meClChargeOnTrack_fpix->setAxisTitle("Charge size (in ke)",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "charge_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "Charge (on track, layer" << i << ")";
    meClChargeOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClChargeOnTrack_layers.at(i-1)->setAxisTitle("Charge size (in ke)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "charge_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "Charge (on track, diskp" << i << ")";
    meClChargeOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClChargeOnTrack_diskps.at(i-1)->setAxisTitle("Charge size (in ke)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "charge_" + clustersrc_.label() + "_Disk_m" << i;
    ss2.str(std::string()); ss2 << "Charge (on track, diskm" << i << ")";
    meClChargeOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClChargeOnTrack_diskms.at(i-1)->setAxisTitle("Charge size (in ke)",1);
  }
  //off track
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OffTrack");
  meClChargeNotOnTrack_all = iBooker.book1D("charge_" + clustersrc_.label(),"Charge (off track)",500,0.,500.);
  meClChargeNotOnTrack_all->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_bpix = iBooker.book1D("charge_" + clustersrc_.label() + "_Barrel","Charge (off track, barrel)",500,0.,500.);
  meClChargeNotOnTrack_bpix->setAxisTitle("Charge size (in ke)",1);
  meClChargeNotOnTrack_fpix = iBooker.book1D("charge_" + clustersrc_.label() + "_Endcap","Charge (off track, endcap)",500,0.,500.);
  meClChargeNotOnTrack_fpix->setAxisTitle("Charge size (in ke)",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "charge_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "Charge (off track, layer" << i << ")";
    meClChargeNotOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClChargeNotOnTrack_layers.at(i-1)->setAxisTitle("Charge size (in ke)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "charge_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "Charge (off track, diskp" << i << ")";
    meClChargeNotOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClChargeNotOnTrack_diskps.at(i-1)->setAxisTitle("Charge size (in ke)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "charge_" + clustersrc_.label() + "_Disk_m" << i;
    ss2.str(std::string()); ss2 << "Charge (off track, diskm" << i << ")";
    meClChargeNotOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClChargeNotOnTrack_diskms.at(i-1)->setAxisTitle("Charge size (in ke)",1);
  }

  //size
  //on track
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OnTrack");
  meClSizeOnTrack_all = iBooker.book1D("size_" + clustersrc_.label(),"Size (on track)",100,0.,100.);
  meClSizeOnTrack_all->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_bpix = iBooker.book1D("size_" + clustersrc_.label() + "_Barrel","Size (on track, barrel)",100,0.,100.);
  meClSizeOnTrack_bpix->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeOnTrack_fpix = iBooker.book1D("size_" + clustersrc_.label() + "_Endcap","Size (on track, endcap)",100,0.,100.);
  meClSizeOnTrack_fpix->setAxisTitle("Cluster size (in pixels)",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "size_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "Size (on track, layer" << i << ")";
    meClSizeOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeOnTrack_layers.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "size_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "Size (on track, diskp" << i << ")";
    meClSizeOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeOnTrack_diskps.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "size_" + clustersrc_.label() + "_Disk_m1" << i;
    ss2.str(std::string()); ss2 << "Size (on track, diskm" << i << ")";
    meClSizeOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeOnTrack_diskms.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  meClSizeXOnTrack_all = iBooker.book1D("sizeX_" + clustersrc_.label(),"SizeX (on track)",100,0.,100.);
  meClSizeXOnTrack_all->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXOnTrack_bpix = iBooker.book1D("sizeX_" + clustersrc_.label() + "_Barrel","SizeX (on track, barrel)",100,0.,100.);
  meClSizeXOnTrack_bpix->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXOnTrack_fpix = iBooker.book1D("sizeX_" + clustersrc_.label() + "_Endcap","SizeX (on track, endcap)",100,0.,100.);
  meClSizeXOnTrack_fpix->setAxisTitle("Cluster sizeX (in pixels)",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "sizeX_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "SizeX (on track, layer" << i << ")";
    meClSizeXOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeXOnTrack_layers.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "sizeX_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "SizeX (on track, diskp" << i << ")";
    meClSizeXOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeXOnTrack_diskps.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "sizeX_" + clustersrc_.label() + "_Disk_m" << i;
    ss2.str(std::string()); ss2 << "SizeX (on track, diskm" << i << ")";
    meClSizeXOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeXOnTrack_diskms.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  meClSizeYOnTrack_all = iBooker.book1D("sizeY_" + clustersrc_.label(),"SizeY (on track)",100,0.,100.);
  meClSizeYOnTrack_all->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYOnTrack_bpix = iBooker.book1D("sizeY_" + clustersrc_.label() + "_Barrel","SizeY (on track, barrel)",100,0.,100.);
  meClSizeYOnTrack_bpix->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYOnTrack_fpix = iBooker.book1D("sizeY_" + clustersrc_.label() + "_Endcap","SizeY (on track, endcap)",100,0.,100.);
  meClSizeYOnTrack_fpix->setAxisTitle("Cluster sizeY (in pixels)",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "sizeY_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "SizeY (on track, layer" << i << ")";
    meClSizeYOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeYOnTrack_layers.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "sizeY_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "SizeY (on track, diskp" << i << ")";
    meClSizeYOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeYOnTrack_diskps.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "sizeY_" + clustersrc_.label() + "_Disk_m" << i;
    ss2.str(std::string()); ss2 << "SizeY (on track, diskm" << i << ")";
    meClSizeYOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeYOnTrack_diskms.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  //off track
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OffTrack");
  meClSizeNotOnTrack_all = iBooker.book1D("size_" + clustersrc_.label(),"Size (off track)",100,0.,100.);
  meClSizeNotOnTrack_all->setAxisTitle("Cluster size (in pixels)",1); 
  meClSizeNotOnTrack_bpix = iBooker.book1D("size_" + clustersrc_.label() + "_Barrel","Size (off track, barrel)",100,0.,100.);
  meClSizeNotOnTrack_bpix->setAxisTitle("Cluster size (in pixels)",1);
  meClSizeNotOnTrack_fpix = iBooker.book1D("size_" + clustersrc_.label() + "_Endcap","Size (off track, endcap)",100,0.,100.);
  meClSizeNotOnTrack_fpix->setAxisTitle("Cluster size (in pixels)",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "size_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "Size (off track, layer" << i << ")";
    meClSizeNotOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeNotOnTrack_layers.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }

  for (int i = 1; i <= noOfLayers; i++) {
    int ybins = -1; float ymin = 0.; float ymax = 0.;
    if (i==1) { ybins = 42; ymin = -10.5; ymax = 10.5; }
    if (i==2) { ybins = 66; ymin = -16.5; ymax = 16.5; }
    if (i==3) { ybins = 90; ymin = -22.5; ymax = 22.5; }
    ss1.str(std::string()); ss1 << "pix_bar Occ_roc_offtrack" + digisrc_. label() + "_layer_" << i;
    ss2.str(std::string()); ss2 << "Pixel Barrel Occupancy, ROC level (Off Track): Layer " << i;
    meZeroRocLadvsModOffTrackBarrel.push_back(iBooker.book2D(ss1.str(),ss2.str(),72,-4.5,4.5,ybins,ymin,ymax));
    meZeroRocLadvsModOffTrackBarrel.at(i-1)->setAxisTitle("ROC / Module",1);
    meZeroRocLadvsModOffTrackBarrel.at(i-1)->setAxisTitle("ROC / Ladder",2);
  }

  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "size_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "Size (off track, diskp" << i << ")";
    meClSizeNotOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeNotOnTrack_diskps.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "size_" + clustersrc_.label() + "_Disk_m" << i;
    ss2.str(std::string()); ss2 << "Size (off track, diskm" << i << ")";
    meClSizeNotOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeNotOnTrack_diskms.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  meClSizeXNotOnTrack_all = iBooker.book1D("sizeX_" + clustersrc_.label(),"SizeX (off track)",100,0.,100.);
  meClSizeXNotOnTrack_all->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXNotOnTrack_bpix = iBooker.book1D("sizeX_" + clustersrc_.label() + "_Barrel","SizeX (off track, barrel)",100,0.,100.);
  meClSizeXNotOnTrack_bpix->setAxisTitle("Cluster sizeX (in pixels)",1);
  meClSizeXNotOnTrack_fpix = iBooker.book1D("sizeX_" + clustersrc_.label() + "_Endcap","SizeX (off track, endcap)",100,0.,100.);
  meClSizeXNotOnTrack_fpix->setAxisTitle("Cluster sizeX (in pixels)",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "sizeX_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "SizeX (off track, layer" << i << ")";
    meClSizeXNotOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeXNotOnTrack_layers.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "sizeX_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "SizeX (off track, diskp" << i << ")";
    meClSizeXNotOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeXNotOnTrack_diskps.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "sizeX_" + clustersrc_.label() + "_Disk_m" << i;
    ss2.str(std::string()); ss2 << "SizeX (off track, diskm" << i << ")";
    meClSizeXNotOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeXNotOnTrack_diskms.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  meClSizeYNotOnTrack_all = iBooker.book1D("sizeY_" + clustersrc_.label(),"SizeY (off track)",100,0.,100.);
  meClSizeYNotOnTrack_all->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYNotOnTrack_bpix = iBooker.book1D("sizeY_" + clustersrc_.label() + "_Barrel","SizeY (off track, barrel)",100,0.,100.);
  meClSizeYNotOnTrack_bpix->setAxisTitle("Cluster sizeY (in pixels)",1);
  meClSizeYNotOnTrack_fpix = iBooker.book1D("sizeY_" + clustersrc_.label() + "_Endcap","SizeY (off track, endcap)",100,0.,100.);
  meClSizeYNotOnTrack_fpix->setAxisTitle("Cluster sizeY (in pixels)",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "sizeY_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "SizeY (off track, layer" << i << ")";
    meClSizeYNotOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeYNotOnTrack_layers.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "sizeY_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "SizeY (off track, diskp" << i << ")";
    meClSizeYNotOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeYNotOnTrack_diskps.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "sizeY_" + clustersrc_.label() + "_Disk_m" << i;
    ss2.str(std::string()); ss2 << "SizeY (off track, diskm" << i << ")";
    meClSizeYNotOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meClSizeYNotOnTrack_diskms.at(i-1)->setAxisTitle("Cluster size (in pixels)",1);
  }

  //cluster global position
  //on track
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OnTrack");
  //bpix
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "position_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "Clusters Layer" << i << " (on track)";
    meClPosLayersOnTrack.push_back(iBooker.book2D(ss1.str(),ss2.str(),200,-30.,30.,128,-3.2,3.2));
    meClPosLayersOnTrack.at(i-1)->setAxisTitle("Global Z (cm)",1);
    meClPosLayersOnTrack.at(i-1)->setAxisTitle("Global #phi",2);

    int ybins = -1; float ymin = 0.; float ymax = 0.;
    if (i==1) { ybins = 23; ymin = -11.5; ymax = 11.5; }
    if (i==2) { ybins = 33; ymin = -17.5; ymax = 17.5; }
    if (i==3) { ybins = 45; ymin = -24.5; ymax = 24.5; }
    ss1.str(std::string()); ss1 << "position_" + clustersrc_.label() + "_LadvsMod_Layer_" << i;
    ss2.str(std::string()); ss2 << "Clusters Layer" << i << "_LadvsMod (on track)";
    meClPosLayersLadVsModOnTrack.push_back(iBooker.book2D(ss1.str(),ss2.str(),11,-5.5,5.5,ybins,ymin,ymax));
    meClPosLayersLadVsModOnTrack.at(i-1)->setAxisTitle("z-module",1);
    meClPosLayersLadVsModOnTrack.at(i-1)->setAxisTitle("Ladder",2);
  }

  for (int i = 1; i <= noOfLayers; i++) {
    int ybins = -1; float ymin = 0.; float ymax = 0.;
    if (i==1) { ybins = 42; ymin = -10.5; ymax = 10.5; }
    if (i==2) { ybins = 66; ymin = -16.5; ymax = 16.5; }
    if (i==3) { ybins = 90; ymin = -22.5; ymax = 22.5; }
    ss1.str(std::string()); ss1 << "pix_bar Occ_roc_ontrack" + digisrc_. label() + "_layer_" << i;
    ss2.str(std::string()); ss2 << "Pixel Barrel Occupancy, ROC level (On Track): Layer " << i;
    meZeroRocLadvsModOnTrackBarrel.push_back(iBooker.book2D(ss1.str(),ss2.str(),72,-4.5,4.5,ybins,ymin,ymax));
    meZeroRocLadvsModOnTrackBarrel.at(i-1)->setAxisTitle("ROC / Module",1);
    meZeroRocLadvsModOnTrackBarrel.at(i-1)->setAxisTitle("ROC / Ladder",2);
  }

  //fpix
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "position_" + clustersrc_.label() + "_pz_Disk_" << i;
    ss2.str(std::string()); ss2 << "Clusters +Z Disk" << i << " (on track)";
    meClPosDiskspzOnTrack.push_back(iBooker.book2D(ss1.str(),ss2.str(),80,-20.,20.,80,-20.,20.));
    meClPosDiskspzOnTrack.at(i-1)->setAxisTitle("Global X (cm)",1);
    meClPosDiskspzOnTrack.at(i-1)->setAxisTitle("Global Y (cm)",2);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "position_" + clustersrc_.label() + "_mz_Disk_" << i;
    ss2.str(std::string()); ss2 << "Clusters -Z Disk" << i << " (on track)";
    meClPosDisksmzOnTrack.push_back(iBooker.book2D(ss1.str(),ss2.str(),80,-20.,20.,80,-20.,20.));
    meClPosDisksmzOnTrack.at(i-1)->setAxisTitle("Global X (cm)",1);
    meClPosDisksmzOnTrack.at(i-1)->setAxisTitle("Global Y (cm)",2);
  }
  meNClustersOnTrack_all = iBooker.book1D("nclusters_" + clustersrc_.label(),"Number of Clusters (on Track)",50,0.,50.);
  meNClustersOnTrack_all->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_bpix = iBooker.book1D("nclusters_" + clustersrc_.label() + "_Barrel","Number of Clusters (on track, barrel)",50,0.,50.);
  meNClustersOnTrack_bpix->setAxisTitle("Number of Clusters",1);
  meNClustersOnTrack_fpix = iBooker.book1D("nclusters_" + clustersrc_.label() + "_Endcap","Number of Clusters (on track, endcap)",50,0.,50.);
  meNClustersOnTrack_fpix->setAxisTitle("Number of Clusters",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "nclusters_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "Number of Clusters (on track, layer" << i << ")";
    meNClustersOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meNClustersOnTrack_layers.at(i-1)->setAxisTitle("Number of Clusters",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "nclusters_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "Number of Clusters (on track, diskp" << i << ")";
    meNClustersOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),50,0.,50.));
    meNClustersOnTrack_diskps.at(i-1)->setAxisTitle("Number of Clusters",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "nclusters_" + clustersrc_.label() + "_Disk_m" << i;
    ss2.str(std::string()); ss2 << "Number of Clusters (on track, diskm" << i << ")";
    meNClustersOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meNClustersOnTrack_diskms.at(i-1)->setAxisTitle("Number of Clusters",1);
  }

  //not on track
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OffTrack");
  //bpix
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "position_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "Clusters Layer" << i << " (off track)";
    meClPosLayersNotOnTrack.push_back(iBooker.book2D(ss1.str(),ss2.str(),200,-30.,30.,128,-3.2,3.2));
    meClPosLayersNotOnTrack.at(i-1)->setAxisTitle("Global Z (cm)",1);
    meClPosLayersNotOnTrack.at(i-1)->setAxisTitle("Global #phi",2);
  }
  //fpix
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "position_" + clustersrc_.label() + "_pz_Disk_" << i;
    ss2.str(std::string()); ss2 << "Clusters +Z Disk" << i << " (off track)";
    meClPosDiskspzNotOnTrack.push_back(iBooker.book2D(ss1.str(),ss2.str(),80,-20.,20.,80,-20.,20.));
    meClPosDiskspzNotOnTrack.at(i-1)->setAxisTitle("Global X (cm)",1);
    meClPosDiskspzNotOnTrack.at(i-1)->setAxisTitle("Global Y (cm)",2);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "position_" + clustersrc_.label() + "_mz_Disk_" << i;
    ss2.str(std::string()); ss2 << "Clusters -Z Disk" << i << " (off track)";
    meClPosDisksmzNotOnTrack.push_back(iBooker.book2D(ss1.str(),ss2.str(),80,-20.,20.,80,-20.,20.));
    meClPosDisksmzNotOnTrack.at(i-1)->setAxisTitle("Global X (cm)",1);
    meClPosDisksmzNotOnTrack.at(i-1)->setAxisTitle("Global Y (cm)",2);
  }
  meNClustersNotOnTrack_all = iBooker.book1D("nclusters_" + clustersrc_.label(),"Number of Clusters (off Track)",50,0.,50.);
  meNClustersNotOnTrack_all->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_bpix = iBooker.book1D("nclusters_" + clustersrc_.label() + "_Barrel","Number of Clusters (off track, barrel)",50,0.,50.);
  meNClustersNotOnTrack_bpix->setAxisTitle("Number of Clusters",1);
  meNClustersNotOnTrack_fpix = iBooker.book1D("nclusters_" + clustersrc_.label() + "_Endcap","Number of Clusters (off track, endcap)",50,0.,50.);
  meNClustersNotOnTrack_fpix->setAxisTitle("Number of Clusters",1);
  for (int i = 1; i <= noOfLayers; i++)
  {
    ss1.str(std::string()); ss1 << "nclusters_" + clustersrc_.label() + "_Layer_" << i;
    ss2.str(std::string()); ss2 << "Number of Clusters (off track, layer" << i << ")";
    meNClustersNotOnTrack_layers.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meNClustersNotOnTrack_layers.at(i-1)->setAxisTitle("Number of Clusters",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "nclusters_" + clustersrc_.label() + "_Disk_p" << i;
    ss2.str(std::string()); ss2 << "Number of Clusters (off track, diskp" << i << ")";
    meNClustersNotOnTrack_diskps.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meNClustersNotOnTrack_diskps.at(i-1)->setAxisTitle("Number of Clusters",1);
  }
  for (int i = 1; i <= noOfDisks; i++)
  {
    ss1.str(std::string()); ss1 << "nclusters_" + clustersrc_.label() + "_Disk_m" << i;
    ss2.str(std::string()); ss2 << "Number of Clusters (off track, diskm" << i << ")";
    meNClustersNotOnTrack_diskms.push_back(iBooker.book1D(ss1.str(),ss2.str(),500,0.,500.));
    meNClustersNotOnTrack_diskms.at(i-1)->setAxisTitle("Number of Clusters",1);
  
  }
  //HitProbability
  //on track
  iBooker.setCurrentFolder(topFolderName_+"/Clusters/OnTrack");
  meHitProbability = iBooker.book1D("FractionLowProb","Fraction of hits with low probability;FractionLowProb;#HitsOnTrack",100,0.,1.);

  if (debug_) {
    // book summary residual histograms in a debugging folder - one (x,y) pair of histograms per subdetector 
    iBooker.setCurrentFolder("debugging"); 
    char hisID[80]; 
    for (int s=0; s<3; s++) {
      sprintf(hisID,"residual_x_subdet_%i",s); 
      meSubdetResidualX[s] = iBooker.book1D(hisID,"Pixel Hit-to-Track Residual in X",500,-5.,5.);
      
      sprintf(hisID,"residual_y_subdet_%i",s);
      meSubdetResidualY[s] = iBooker.book1D(hisID,"Pixel Hit-to-Track Residual in Y",500,-5.,5.);  
    }
  }
  
  firstRun = false;
}


void SiPixelTrackResidualSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  
  // retrieve TrackerGeometry again and MagneticField for use in transforming 
  // a TrackCandidate's P(ersistent)TrajectoryStateoOnDet (PTSoD) to a TrajectoryStateOnSurface (TSoS)
  ESHandle<TrackerGeometry> TG;
  iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  const TrackerGeometry* theTrackerGeometry = TG.product();
  
  //analytic triplet method to calculate the track residuals in the pixe barrel detector    
  
  //--------------------------------------------------------------------
  // beam spot:
  //
  edm::Handle<reco::BeamSpot> rbs;
  //iEvent.getByLabel( "offlineBeamSpot", rbs );
  iEvent.getByToken( beamSpotToken_, rbs );
  math::XYZPoint bsP = math::XYZPoint(0,0,0);
  if( !rbs.failedToGet() && rbs.isValid() )
    {
      bsP = math::XYZPoint( rbs->x0(), rbs->y0(), rbs->z0() );
    }
  
  //--------------------------------------------------------------------
  // primary vertices:
  //
  edm::Handle<reco::VertexCollection> vertices;
  //iEvent.getByLabel("offlinePrimaryVertices", vertices );
  iEvent.getByToken(  offlinePrimaryVerticesToken_, vertices );
  
  if( vertices.failedToGet() ) return;
  if( !vertices.isValid() ) return;
  
  math::XYZPoint vtxN = math::XYZPoint(0,0,0);
  math::XYZPoint vtxP = math::XYZPoint(0,0,0);
  
  double bestNdof = 0.0;
  double maxSumPt = 0.0;
  reco::Vertex bestPvx;
  for(reco::VertexCollection::const_iterator iVertex = vertices->begin();
      iVertex != vertices->end(); ++iVertex ) {
    if( iVertex->ndof() > bestNdof ) {
      bestNdof = iVertex->ndof();
      vtxN = math::XYZPoint( iVertex->x(), iVertex->y(), iVertex->z() );
    }//ndof
    if( iVertex->p4().pt() > maxSumPt ) {
      maxSumPt = iVertex->p4().pt();
      vtxP = math::XYZPoint( iVertex->x(), iVertex->y(), iVertex->z() );
      bestPvx = *iVertex;
    }//sumpt
    
  }//vertex
  
  if( maxSumPt < 1.0 ) vtxP = vtxN;
  
  //---------------------------------------------
  //get Tracks
  //
  edm:: Handle<reco::TrackCollection> TracksForRes;
  //iEvent.getByLabel( "generalTracks", TracksForRes );
  iEvent.getByToken(  generalTracksToken_, TracksForRes );

  //
  // transient track builder, needs B-field from data base (global tag in .py)
  //
  edm::ESHandle<TransientTrackBuilder> theB;
  iSetup.get<TransientTrackRecord>().get( "TransientTrackBuilder", theB );

  //get the TransienTrackingRecHitBuilder needed for extracting the global position of the hits in the pixel
  edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;
  iSetup.get<TransientRecHitRecord>().get(ttrhbuilder_,theTrackerRecHitBuilder);

  //check that tracks are valid
  if( TracksForRes.failedToGet() ) return;
  if( !TracksForRes.isValid() ) return;

  //get tracker geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  
  if( !pDD.isValid() ) {
    cout << "Unable to find TrackerDigiGeometry. Return\n";
    return;
  }
 
  int kk = -1;
  //----------------------------------------------------------------------------
  // Residuals:
  //
  for( reco::TrackCollection::const_iterator iTrack = TracksForRes->begin();
       iTrack != TracksForRes->end(); ++iTrack ) {
    //count
    kk++;
    //Calculate minimal track pt before curling
    // cpt = cqRB = 0.3*R[m]*B[T] = 1.14*R[m] for B=3.8T
    // D = 2R = 2*pt/1.14
    // calo: D = 1.3 m => pt = 0.74 GeV/c
    double pt = iTrack->pt();
    if( pt < 0.75 ) continue;// curls up
    if( abs( iTrack->dxy(vtxP) ) > 5*iTrack->dxyError() ) continue; // not prompt
    
    double charge = iTrack->charge();
       
    reco::TransientTrack tTrack = theB->build(*iTrack);
    //get curvature of the track, needed for the residuals
    double kap = tTrack.initialFreeState().transverseCurvature();
    //needed for the TransienTrackingRecHitBuilder
    TrajectoryStateOnSurface initialTSOS = tTrack.innermostMeasurementState();
    if( iTrack->extra().isNonnull() &&iTrack->extra().isAvailable() ){
      
      double x1 = 0;
      double y1 = 0;
      double z1 = 0;
      double x2 = 0;
      double y2 = 0;
      double z2 = 0;
      double x3 = 0;
      double y3 = 0;
      double z3 = 0;
      int n1 = 0;
      int n2 = 0;
      int n3 = 0;
      
      //for saving the pixel barrel hits
      vector<TransientTrackingRecHit::RecHitPointer> GoodPixBarrelHits;
      //looping through the RecHits of the track
      for( trackingRecHit_iterator irecHit = iTrack->recHitsBegin();
	   irecHit != iTrack->recHitsEnd(); ++irecHit){
	
	if( (*irecHit)->isValid() ){
	  DetId detId = (*irecHit)->geographicalId();
	  // enum Detector { Tracker=1, Muon=2, Ecal=3, Hcal=4, Calo=5 };
	  if( detId.det() != 1 ){
	    if(debug_){
	      cout << "rec hit ID = " << detId.det() << " not in tracker!?!?\n";
	    }
	    continue;
	  }
	  uint32_t subDet = detId.subdetId();
	  
	  // enum SubDetector{ PixelBarrel=1, PixelEndcap=2 };
	  // enum SubDetector{ TIB=3, TID=4, TOB=5, TEC=6 };
	  
	  TransientTrackingRecHit::RecHitPointer trecHit = theTrackerRecHitBuilder->build(  &*(*irecHit), initialTSOS);
	  
	  
	  double gX = trecHit->globalPosition().x();
	  double gY = trecHit->globalPosition().y();
	  double gZ = trecHit->globalPosition().z();
	      
	      
	      if( subDet == PixelSubdetector::PixelBarrel ) {

		int ilay = tTopo->pxbLayer(detId);
		
		if( ilay == 1 ){
		  n1++;
		  x1 = gX;
		  y1 = gY;
		  z1 = gZ;
		  
		  GoodPixBarrelHits.push_back((trecHit));
		}//PXB1
		if( ilay == 2 ){
		  
		  n2++;
		  x2 = gX;
		  y2 = gY;
		  z2 = gZ;
		  
		  GoodPixBarrelHits.push_back((trecHit));
		  
		}//PXB2
		if( ilay == 3 ){
		  
		  n3++;
		  x3 = gX;
		  y3 = gY;
		  z3 = gZ;
		  GoodPixBarrelHits.push_back((trecHit));
		}
	      }//PXB
	      
	      
	    }//valid
	  }//loop rechits
	  
	  //CS extra plots    
	  
	 	  
	  if( n1+n2+n3 == 3 && n1*n2*n3 > 0) {      
	    for( unsigned int i = 0; i < GoodPixBarrelHits.size(); i++){
	      
	      if( GoodPixBarrelHits[i]->isValid() ){
		DetId detId = GoodPixBarrelHits[i]->geographicalId().rawId();
		int ilay = tTopo->pxbLayer(detId);
		if(pt > ptminres_){   
		  
		  double dca2 = 0.0, dz2=0.0;
		  double ptsig = pt;
		  if(charge<0.) ptsig = -pt;
		  //Filling the histograms in modules
		  
		  MeasurementPoint Test;
		  MeasurementPoint Test2;
		  Test=MeasurementPoint(0,0);
		  Test2=MeasurementPoint(0,0);
		  Measurement2DVector residual;
		 
		  if( ilay == 1 ){
		    
		    triplets(x2,y2,z2,x1,y1,z1,x3,y3,z3,ptsig,dca2,dz2, kap);	
		    		
		    Test=MeasurementPoint(dca2*1E4,dz2*1E4);
		    residual=Test-Test2;
		  }
		  
		  if( ilay == 2 ){
		    
		    triplets(x1,y1,z1,x2,y2,z2,x3,y3,z3,ptsig,dca2,dz2, kap);
		    
		    Test=MeasurementPoint(dca2*1E4,dz2*1E4);
		    residual=Test-Test2;
	      
		  }
		  
		  if( ilay == 3 ){
		    
		    triplets(x1,y1,z1,x3,y3,z3,x2,y2,z2,ptsig,dca2,dz2, kap);
		    
		    Test=MeasurementPoint(dca2*1E4,dz2*1E4);
		    residual=Test-Test2;
	    }
		  // fill the residual histograms 

		  std::map<uint32_t, SiPixelTrackResidualModule*>::iterator pxd = theSiPixelStructure.find(detId);
		  if (pxd!=theSiPixelStructure.end()) (*pxd).second->fill(residual, reducedSet, modOn, ladOn, layOn, phiOn, bladeOn, diskOn, ringOn);	

		  if(ladOn&&pxd!=theSiPixelStructure.end()){  
		    meResidualXSummedLay.at(PixelBarrelNameUpgrade((*pxd).first).layerName()-1)->Fill(residual.x());    
		    meResidualYSummedLay.at(PixelBarrelNameUpgrade((*pxd).first).layerName()-1)->Fill(residual.y());    
		  }
	
		}//three hits
	      }//is valid
	    }//rechits loop
	  }//pt 4
	}
	
	
 }//-----Tracks
  ////////////////////////////
  //get trajectories
  edm::Handle<std::vector<Trajectory> > trajCollectionHandle;
  //iEvent.getByLabel(tracksrc_,trajCollectionHandle);
  iEvent.getByToken ( tracksrcToken_, trajCollectionHandle );
  auto const & trajColl = *(trajCollectionHandle.product());
   
  //get tracks
  edm::Handle<std::vector<reco::Track> > trackCollectionHandle;
  //iEvent.getByLabel(tracksrc_,trackCollectionHandle);
  iEvent.getByToken( trackToken_, trackCollectionHandle );
  auto const & trackColl = *(trackCollectionHandle.product());
  
  //get the map
  edm::Handle<TrajTrackAssociationCollection> match;
  //iEvent.getByLabel(tracksrc_,match);
  iEvent.getByToken( trackAssociationToken_, match);
  auto const &  ttac = *(match.product());
  
  // get clusters
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> >  clusterColl;
  //iEvent.getByLabel( clustersrc_, clusterColl );
  iEvent.getByToken( clustersrcToken_, clusterColl );
  auto const & clustColl = *(clusterColl.product());

  // get digis
  edm::Handle< edm::DetSetVector<PixelDigi> >  digiinput;
  iEvent.getByToken( digisrcToken_, digiinput );
  const edm::DetSetVector<PixelDigi> diginp = *(digiinput.product());
  

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

    const std::vector<TrajectoryMeasurement>& tmeasColl =traj_iterator->measurements();
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

      const std::vector<TrajectoryMeasurement>& tmeasColl = traj_iterator->measurements();
      for(std::vector<TrajectoryMeasurement>::const_iterator tmeasIt = tmeasColl.begin(); tmeasIt!=tmeasColl.end(); tmeasIt++){   
	if(! tmeasIt->updatedState().isValid()) continue; 
	
	TrajectoryStateOnSurface tsos = tsoscomb( tmeasIt->forwardPredictedState(), tmeasIt->backwardPredictedState() );
        if (!tsos.isValid()) continue; // Happens rarely, due to singular matrix or similar

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

		DetId detId = (*hit).geographicalId();
		if(detId>=302055684 && detId<=352477708) { getrococcupancy(detId,diginp,tTopo,meZeroRocLadvsModOnTrackBarrel); }

		//CORR CHARGE
		meClChargeOnTrack_bpix->Fill(corrCharge);
		meClSizeOnTrack_bpix->Fill((*clust).size());
		meClSizeXOnTrack_bpix->Fill((*clust).sizeX());
		meClSizeYOnTrack_bpix->Fill((*clust).sizeY());
		int DBlayer;
		DBlayer = PixelBarrelName(DetId((*hit).geographicalId()), tTopo, isUpgrade).layerName();
		float phi = clustgp.phi(); 
		float z = clustgp.z();

                PixelBarrelName pbn(DetId((*hit).geographicalId()), tTopo, isUpgrade);
                int ladder = pbn.ladderName();
                int module = pbn.moduleName();

		PixelBarrelName::Shell sh = pbn.shell(); //enum
                int ladderSigned=ladder;
                int moduleSigned=module;
                // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
                int shell = int(sh);
                // change the module sign for z<0
                if(shell==1 || shell==2) { moduleSigned = -module; }
                // change ladeer sign for Outer )x<0)
                if(shell==1 || shell==3) { ladderSigned = -ladder; }

		for (int i = 0; i < noOfLayers; i++)
		  {
          if (DBlayer == i + 1) {
             meClPosLayersOnTrack.at(i)->Fill(z,phi);
	     meClPosLayersLadVsModOnTrack.at(i)->Fill(moduleSigned,ladderSigned);
             meClChargeOnTrack_layers.at(i)->Fill(corrCharge);
             meClSizeOnTrack_layers.at(i)->Fill((*clust).size());
             meClSizeXOnTrack_layers.at(i)->Fill((*clust).sizeX());
             meClSizeYOnTrack_layers.at(i)->Fill((*clust).sizeY());
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
		int DBdisk = 0;
		DBdisk = PixelEndcapName(DetId((*hit).geographicalId()), tTopo, isUpgrade).diskName();
		float x = clustgp.x(); 
		float y = clustgp.y(); 
		float z = clustgp.z();
		if(z>0){
        for (int i = 0; i < noOfDisks; i++)
        {
            if (DBdisk == i + 1) {
               meClPosDiskspzOnTrack.at(i)->Fill(x,y);
               meClChargeOnTrack_diskps.at(i)->Fill(corrCharge);
               meClSizeOnTrack_diskps.at(i)->Fill((*clust).size());
               meClSizeXOnTrack_diskps.at(i)->Fill((*clust).sizeX());
               meClSizeYOnTrack_diskps.at(i)->Fill((*clust).sizeY());
            }
        }
		}
		else{
        for (int i = 0; i < noOfDisks; i++)
        {
          if (DBdisk == i + 1) {
             meClPosDisksmzOnTrack.at(i)->Fill(x,y);
             meClChargeOnTrack_diskms.at(i)->Fill(corrCharge);
             meClSizeOnTrack_diskms.at(i)->Fill((*clust).size());
             meClSizeXOnTrack_diskms.at(i)->Fill((*clust).sizeX());
             meClSizeYOnTrack_diskms.at(i)->Fill((*clust).sizeY());
          }
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
    //if(dynamic_cast<PixelGeomDetUnit const *>((*it))!=0){
    DetId detId = (*it)->geographicalId();
    if(detId>=302055684 && detId<=352477708){ // make sure it's a Pixel module WITHOUT using dynamic_cast!  
      int nofclOnTrack = 0, nofclOffTrack=0; 
      float z=0.; 
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
	    bool barrel = DetId(detId).subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel);
	    if (barrel) { getrococcupancy(detId,diginp,tTopo,meZeroRocLadvsModOffTrackBarrel); }

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
	      int DBlayer = PixelBarrelName(DetId(detId), tTopo, isUpgrade).layerName();
	      float phi = clustgp.phi(); 
	      //float r = clustgp.perp();
	      z = clustgp.z();
         for (int i = 0; i < noOfLayers; i++)
         {
           if (DBlayer == i + 1) {
              meClPosLayersNotOnTrack.at(i)->Fill(z,phi);
              meClSizeNotOnTrack_layers.at(i)->Fill((*di).size());
              meClSizeXNotOnTrack_layers.at(i)->Fill((*di).sizeX());
              meClSizeYNotOnTrack_layers.at(i)->Fill((*di).sizeY());
              meClChargeNotOnTrack_layers.at(i)->Fill((*di).charge()/1000);
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
	      int DBdisk = PixelEndcapName(DetId(detId), tTopo, isUpgrade).diskName();
	      float x = clustgp.x(); 
	      float y = clustgp.y(); 
	      z = clustgp.z();
	      if(z>0){
           for (int i = 0; i < noOfDisks; i++)
           {
             if (DBdisk == i + 1) {
                meClPosDiskspzNotOnTrack.at(i)->Fill(x,y);
                meClSizeNotOnTrack_diskps.at(i)->Fill((*di).size());
                meClSizeXNotOnTrack_diskps.at(i)->Fill((*di).sizeX());
                meClSizeYNotOnTrack_diskps.at(i)->Fill((*di).sizeY());
                meClChargeNotOnTrack_diskps.at(i)->Fill((*di).charge()/1000);
             }
           }
	      }
	      else{
           for (int i = 0; i < noOfDisks; i++)
           {
             if (DBdisk == i + 1) {
                meClPosDisksmzNotOnTrack.at(i)->Fill(x,y);
                meClSizeNotOnTrack_diskms.at(i)->Fill((*di).size());
                meClSizeXNotOnTrack_diskms.at(i)->Fill((*di).sizeX());
                meClSizeYNotOnTrack_diskms.at(i)->Fill((*di).sizeY());
                meClChargeNotOnTrack_diskms.at(i)->Fill((*di).charge()/1000);
             }
           }
	      } 

	    }
	  }// end "if cluster off track"
	  else {
	    nofclOnTrack++; 
	    if(z == 0){
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
	int DBlayer = PixelBarrelName(DetId(detId), tTopo, isUpgrade).layerName();
        for (int i = 0; i < noOfLayers; i++)
        {
          if (DBlayer == i + 1) {
             if(nofclOnTrack!=0) meNClustersOnTrack_layers.at(i)->Fill(nofclOnTrack);
             if(nofclOffTrack!=0) meNClustersNotOnTrack_layers.at(i)->Fill(nofclOffTrack);
          }
        }
      }//end barrel
      //endcap
      if(DetId(detId).subdetId() == 2) {
	int DBdisk = PixelEndcapName(DetId(detId )).diskName();
	//z = clustgp.z();
	if(nofclOnTrack!=0) meNClustersOnTrack_fpix->Fill(nofclOnTrack); 
	if(nofclOffTrack!=0) meNClustersNotOnTrack_fpix->Fill(nofclOffTrack);
	if(z>0){
     for (int i = 0; i < noOfDisks; i++)
     {
       if (DBdisk == i + 1) {
          if(nofclOnTrack!=0) meNClustersOnTrack_diskps.at(i)->Fill(nofclOnTrack);
          if(nofclOffTrack!=0) meNClustersNotOnTrack_diskps.at(i)->Fill(nofclOffTrack);
       }
     }
	}
	if(z<0){
     for (int i = 0; i < noOfDisks; i++)
     {
       if (DBdisk == i + 1) {
         if(nofclOnTrack!=0) meNClustersOnTrack_diskms.at(i)->Fill(nofclOnTrack);
         if(nofclOffTrack!=0) meNClustersNotOnTrack_diskms.at(i)->Fill(nofclOffTrack);
       }
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

void SiPixelTrackResidualSource::getrococcupancy(DetId detId,const edm::DetSetVector<PixelDigi> diginp,const TrackerTopology* const tTopo,std::vector<MonitorElement*> meinput) {

  edm::DetSetVector<PixelDigi>::const_iterator ipxsearch = diginp.find(detId);
  if( ipxsearch != diginp.end() ) {

    // Look at digis now
    edm::DetSet<PixelDigi>::const_iterator  pxdi;
    for (pxdi = ipxsearch->begin(); pxdi != ipxsearch->end(); pxdi++) {

      bool isHalfModule = PixelBarrelName(DetId(detId),tTopo,isUpgrade).isHalfModule();
      int  DBlayer      = PixelBarrelName(DetId(detId),tTopo,isUpgrade).layerName();
      int  DBmodule     = PixelBarrelName(DetId(detId),tTopo,isUpgrade).moduleName();
      int  DBladder     = PixelBarrelName(DetId(detId),tTopo,isUpgrade).ladderName();
      int  DBshell      = PixelBarrelName(DetId(detId),tTopo,isUpgrade).shell();

      // add sign to the modules
      if (DBshell==1 || DBshell==2) { DBmodule = -DBmodule; }
      if (DBshell==1 || DBshell==3) { DBladder = -DBladder; }

      int col = pxdi->column();
      int row = pxdi->row();

      float modsign = (float)DBmodule/(abs((float)DBmodule));
      float ladsign = (float)DBladder/(abs((float)DBladder));
      float rocx = ((float)col/(52.*8.))*modsign + ((float)DBmodule-(modsign)*0.5);
      float rocy = ((float)row/(80.*2.))*ladsign + ((float)DBladder-(ladsign)*0.5);

      // do the flip where need
      bool flip    = false;
      if ( (DBladder%2==0) && (!isHalfModule) ) { flip = true; }
      if ((flip) && (DBladder>0)) {
        if      ( ( ((float)DBladder-(ladsign)*0.5)<=rocy) && (rocy<(float)DBladder))    { rocy = rocy + ladsign*0.5; }
        else if ( ( ((float)DBladder)<=rocy) && (rocy<((float)DBladder+(ladsign)*0.5)) ) { rocy = rocy - ladsign*0.5; }
      }

      // tweak border effect for negative modules/ladders
      if (modsign<0) { rocx = rocx -0.0001; }
      if (ladsign<0) { rocy = rocy -0.0001; } else { rocy = rocy +0.0001; }
      if (abs(DBladder)==1) { rocy = rocy + ladsign*0.5; } //take care of the half module

      if (DBlayer==1) { meinput.at(0)->Fill(rocx,rocy); }
      if (DBlayer==2) { meinput.at(1)->Fill(rocx,rocy); }
      if (DBlayer==3) { meinput.at(2)->Fill(rocx,rocy); }
    } // end of looping over pxdi
  }
}



void SiPixelTrackResidualSource::triplets(double x1,double y1,double z1,double x2,double y2,double z2,double x3,double y3,double z3,
					  double ptsig, double & dca2,double & dz2, double kap) {
  
  //Define some constants
  using namespace std;

  //Curvature kap from global Track

 //inverse of the curvature is the radius in the transverse plane
  double rho = 1/kap;
  //Check that the hits are in the correct layers
  double r1 = sqrt( x1*x1 + y1*y1 );
  double r3 = sqrt( x3*x3 + y3*y3 );

  if( r3-r1 < 2.0 ) cout << "warn r1 = " << r1 << ", r3 = " << r3 << endl;
  
  // Calculate the centre of the helix in xy-projection with radius rho from the track.
  //start with a line (sekante) connecting the two points (x1,y1) and (x3,y3) vec_L = vec_x3-vec_x1
  //with L being the length of that vector.
  double L=sqrt((x3-x1)*(x3-x1)+(y3-y1)*(y3-y1));
  //lam is the line from the middel point of vec_q towards the center of the circle X0,Y0
  // we already have kap and rho = 1/kap
  double lam = sqrt(rho*rho - L*L/4);
  
  // There are two solutions, the sign of kap gives the information
  // which of them is correct.
  //
  if( kap > 0 ) lam = -lam;

  //
  // ( X0, Y0 ) is the centre of the circle that describes the helix in xy-projection.
  //
  double x0 =  0.5*( x1 + x3 ) + lam/L * ( -y1 + y3 );
  double y0 =  0.5*( y1 + y3 ) + lam/L * (  x1 - x3 );

  // Calculate the dipangle in z direction (needed later for z residual) :
  //Starting from the heliz equation whihc has to hold for both points z1,z3
  double num = ( y3 - y0 ) * ( x1 - x0 ) - ( x3 - x0 ) * ( y1 - y0 );
  double den = ( x1 - x0 ) * ( x3 - x0 ) + ( y1 - y0 ) * ( y3 - y0 );
  double tandip = kap * ( z3 - z1 ) / atan( num / den );

 
  // angle from first hit to dca point:
  //
  double dphi = atan( ( ( x1 - x0 ) * y0 - ( y1 - y0 ) * x0 )
                      / ( ( x1 - x0 ) * x0 + ( y1 - y0 ) * y0 ) );
  //z position of the track based on the middle of the circle
  //track equation for the z component
  double uz0 = z1 + tandip * dphi * rho;

  /////////////////////////
  //RESIDUAL IN R-PHI
  ////////////////////////////////
  //Calculate distance dca2 from point (x2,y2) to the circle which is given by
  //the distance of the point to the middlepoint dcM = sqrt((x0-x2)^2+(y0-y2)) and rho
  //dca = rho +- dcM
  if(kap>0) dca2=rho-sqrt((x0-x2)*(x0-x2)+(y0-y2)*(y0-y2));
  else dca2=rho+sqrt((-x0+x2)*(-x0+x2)+(-y0+y2)*(-y0+y2));

  /////////////////////////
  //RESIDUAL IN Z
  /////////////////////////
  double xx =0 ;
  double yy =0 ;
  //sign of kappa determines the calculation
  //xx and yy are the new coordinates starting from x2, y2 that are on the track itself
  //vec_X2+-dca2*vec(X0-X2)/|(X0-X2)|
  if(kap<0){
    xx =   x2+(dca2*((x0-x2))/sqrt((x0-x2)*(x0-x2)+(y0-y2)*(y0-y2)));
    yy = y2+(dca2*((y0-y2))/sqrt((x0-x2)*(x0-x2)+(y0-y2)*(y0-y2)));
  }
  else if(kap>=0){
    xx =   x2-(dca2*((x0-x2))/sqrt((x0-x2)*(x0-x2)+(y0-y2)*(y0-y2)));
    yy = y2-(dca2*((y0-y2))/sqrt((x0-x2)*(x0-x2)+(y0-y2)*(y0-y2)));
  }

  //to get residual in z start with calculating the new uz2 position if one has moved to xx, yy
  //on the track. First calculate the change in phi2 with respect to the center X0, Y0
  double dphi2 = atan( ( ( xx - x0 ) * y0 - ( yy - y0 ) * x0 )
		       / ( ( xx - x0 ) * x0 + ( yy - y0 ) * y0 ) );
  //Solve track equation for this new z depending on the dip angle of the track (see above
  //calculated based on X1, X3 and X0, use uz0 as reference point again.
  double  uz2= uz0 - dphi2*tandip*rho;
  
  //subtract new z position from the old one
  dz2=z2-uz2;
  
  //if we are interested in the arclength this is unsigned though
  //  double cosphi2 = (x2*xx+y2*yy)/(sqrt(x2*x2+y2*y2)*sqrt(xx*xx+yy*yy));
  //double arcdca2=sqrt(x2*x2+y2*y2)*acos(cosphi2);
  
}


DEFINE_FWK_MODULE(SiPixelTrackResidualSource); // define this as a plug-in 
