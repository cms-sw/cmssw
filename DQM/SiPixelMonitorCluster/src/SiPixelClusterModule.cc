// -*- C++ -*-
//
// Package:    SiPixelMonitorCluster
// Class:      SiPixelClusterSource
// 
/**\class 

 Description: Pixel DQM source for Clusters

 Implementation:
     Note that the x- and y-directions referred to in the cluster description refer to local x- and y-values given by the clusterizer.  Local x corresponds to row value and local y corresponds to column value.
*/
//
// Original Author:  Vincenzo Chiochia & Andrew York
//         Created:  
// $Id: SiPixelClusterModule.cc,v 1.9 2007/05/24 17:55:09 andrewdc Exp $
//
//
#include "DQM/SiPixelMonitorCluster/interface/SiPixelClusterModule.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiPixelCommon/interface/SiPixelHistogramId.h"
/// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <stdlib.h>
//
// Constructors
//
SiPixelClusterModule::SiPixelClusterModule() : id_(0),
					 ncols_(416),
					 nrows_(160) { }
///
SiPixelClusterModule::SiPixelClusterModule(const uint32_t& id) : 
  id_(id),
  ncols_(416),
  nrows_(160)
{ 
}
///
SiPixelClusterModule::SiPixelClusterModule(const uint32_t& id, const int& ncols, const int& nrows) : 
  id_(id),
  ncols_(ncols),
  nrows_(nrows)
{ 
}
//
// Destructor
//
SiPixelClusterModule::~SiPixelClusterModule() {}
//
// Book histograms
//
void SiPixelClusterModule::book(const edm::ParameterSet& iConfig) {

  std::string hid;
  // Get collection name and instantiate Histo Id builder
  edm::InputTag src = iConfig.getParameter<edm::InputTag>( "src" );
  SiPixelHistogramId* theHistogramId = new SiPixelHistogramId( src.label() );
  // Get DQM interface
  DaqMonitorBEInterface* theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
  // Number of clusters
  hid = theHistogramId->setHistoId("nclusters",id_);
  meNClusters_ = theDMBE->book1D(hid,"Number of Clusters",50,0.,50.);
  meNClusters_->setAxisTitle("Number of Clusters",1);
  // Total cluster charge in MeV
  hid = theHistogramId->setHistoId("charge",id_);
  meCharge_ = theDMBE->book1D(hid,"Cluster charge",500,0.,500.);
  meCharge_->setAxisTitle("Charge size (MeV)",1);
  // Cluster barycenter X position
  hid = theHistogramId->setHistoId("x",id_);
  meX_ = theDMBE->book1D(hid,"Cluster barycenter X (row #)",200,0.,200.);
  meX_->setAxisTitle("Barycenter x-position (row #)",1);
  // Cluster barycenter Y position
  hid = theHistogramId->setHistoId("y",id_);
  meY_ = theDMBE->book1D(hid,"Cluster barycenter Y (column #)",500,0.,500.);
  meY_->setAxisTitle("Barycenter y-position (column #)",1);
  // Total cluster size (in pixels)
  hid = theHistogramId->setHistoId("size",id_);
  meSize_ = theDMBE->book1D(hid,"Total cluster size",100,0.,100.);
  meSize_->setAxisTitle("Cluster size (in pixels)",1);
  // Cluster width on the x-axis
  hid = theHistogramId->setHistoId("sizeX",id_);
  meSizeX_ = theDMBE->book1D(hid,"Cluster x-width (rows)",10,0.,10.);
  meSizeX_->setAxisTitle("Cluster x-size (rows)",1);
  // Cluster width on the y-axis
  hid = theHistogramId->setHistoId("sizeY",id_);
  meSizeY_ = theDMBE->book1D(hid,"Cluster y-width (columns)",20,0.,20.);
  meSizeY_->setAxisTitle("Cluster y-size (columns)",1);
  // Lowest cluster row
  hid = theHistogramId->setHistoId("minrow",id_);
  meMinRow_ = theDMBE->book1D(hid,"Lowest cluster row",200,0.,200.);
  meMinRow_->setAxisTitle("Lowest cluster row",1);
  // Highest cluster row
  hid = theHistogramId->setHistoId("maxrow",id_);
  meMaxRow_ = theDMBE->book1D(hid,"Highest cluster row",200,0.,200.);
  meMaxRow_->setAxisTitle("Highest cluster row",1);
  // Lowest cluster column
  hid = theHistogramId->setHistoId("mincol",id_);
  meMinCol_ = theDMBE->book1D(hid,"Lowest cluster column",500,0.,500.);
  meMinCol_->setAxisTitle("Lowest cluster column",1);
  // Highest cluster column
  hid = theHistogramId->setHistoId("maxcol",id_);
  meMaxCol_ = theDMBE->book1D(hid,"Highest cluster column",500,0.,500.);
  meMaxCol_->setAxisTitle("Highest cluster column",1);
  // 2D hit map
  int nbinx = ncols_/2;
  int nbiny = nrows_/2;
  hid = theHistogramId->setHistoId("hitmap",id_);
  mePixClusters_ = theDMBE->book2D(hid,"Number of Clusters (1bin=four pixels)",nbinx,0.,float(ncols_),nbiny,0.,float(nrows_));
  mePixClusters_->setAxisTitle("Columns",1);
  mePixClusters_->setAxisTitle("Rows",2);

  delete theHistogramId;
}
//
// Fill histograms
//
void SiPixelClusterModule::fill(const edm::DetSetVector<SiPixelCluster>& input) {
  
  edm::DetSetVector<SiPixelCluster>::const_iterator isearch = input.find(id_); // search  clusters of detid
  
  if( isearch != input.end() ) {  // Not at empty iterator
    
    unsigned int numberOfClusters = 0;
    
    // Look at clusters now
    edm::DetSet<SiPixelCluster>::const_iterator  di;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      numberOfClusters++;
      float charge = 0.001*(di->charge()); // total charge of cluster
      float x = di->x();                   // barycenter x position
      float y = di->y();                   // barycenter y position
      int size = di->size();               // total size of cluster (in pixels)
      int sizeX = di->sizeX();             // size of cluster in x-direction
      int sizeY = di->sizeY();             // size of cluster in y-direction
      int minPixelRow = di->minPixelRow(); // min x index
      int maxPixelRow = di->maxPixelRow(); // max x index
      int minPixelCol = di->minPixelCol(); // min y index
      int maxPixelCol = di->maxPixelCol(); // max y index
      //      bool edgeHitX = di->edgeHitX();      // records if a cluster is at the x-edge of the detector
      //      bool edgeHitY = di->edgeHitY();      // records if a cluster is at the y-edge of the detector

      (meCharge_)->Fill((float)charge);
      (meX_)->Fill((float)x);
      (meY_)->Fill((float)y);
      (meSize_)->Fill((int)size);
      (meSizeX_)->Fill((int)sizeX);
      (meSizeY_)->Fill((int)sizeY);
      (meMinRow_)->Fill((int)minPixelRow);
      (meMaxRow_)->Fill((int)maxPixelRow);
      (meMinCol_)->Fill((int)minPixelCol);
      (meMaxCol_)->Fill((int)maxPixelCol);
      (mePixClusters_)->Fill((float)y,(float)x);
      //      (meEdgeHitX_)->Fill((int)edgeHitX);
      //      (meEdgeHitY_)->Fill((int)edgeHitY);

    }
    (meNClusters_)->Fill((float)numberOfClusters);
    //std::cout<<"number of clusters="<<numberOfClusters<<std::endl;
      
  }
  
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}
