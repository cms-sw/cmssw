#include "DQM/SiPixelMonitorCluster/interface/SiPixelClusterModule.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetId.h"
//#include "DataFormats/SiPixelCluster/interface/PixelClusterCollection.h"
//#include "DataFormats/SiPixelCluster/interface/PixelCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

// Framework
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// STL
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <boost/cstdint.hpp>
#include <string>
#include <stdlib.h>
//
// Constructors
//
SiPixelClusterModule::SiPixelClusterModule() {

}

SiPixelClusterModule::SiPixelClusterModule(uint32_t id): id_(id) { }

//
// Destructor
//
SiPixelClusterModule::~SiPixelClusterModule() {}

//
// Book histograms
//
void SiPixelClusterModule::book() {
  DaqMonitorBEInterface* theDMBE = edm::Service<DaqMonitorBEInterface>().operator->();
  char hkey[80];  
  sprintf(hkey, "nclusters_module_%i",id_);                             
  meNClusters_ = theDMBE->book1D(hkey,"Number of Clusters",50,0.,50.);  
  sprintf(hkey, "y_module_%i",id_);
  meY_ = theDMBE->book1D(hkey,"Cluster(barycenter) Y",500,0.,500.);          
  sprintf(hkey, "x_module_%i",id_);
  meX_ = theDMBE->book1D(hkey,"Cluster(barycenter) X",200,0.,200.);
  sprintf(hkey, "charge_module_%i",id_);
  meCharge_ = theDMBE->book1D(hkey,"Cluster charge",500,0.,500.);  //in MeV   
  sprintf(hkey, "size_module_%i",id_);
  meSize_ = theDMBE->book1D(hkey,"Cluster size (total pixels)",100,0.,100.);
  sprintf(hkey, "sizeX_module_%i",id_);
  meSizeX_ = theDMBE->book1D(hkey,"Cluster x size",10,0.,10.);
  sprintf(hkey, "sizeY_module_%i",id_);
  meSizeY_ = theDMBE->book1D(hkey,"Cluster y size",20,0,20.);
  sprintf(hkey, "minrow_module_%i",id_);
  meMinRow_ = theDMBE->book1D(hkey,"Lowest Cluster row",200,0.,200.);
  sprintf(hkey, "maxrow_module_%i",id_);
  meMaxRow_ = theDMBE->book1D(hkey,"Highest Cluster row",200,0.,200.);
  sprintf(hkey, "mincol_module_%i",id_);
  meMinCol_ = theDMBE->book1D(hkey,"Lowest Cluster column",500,0.,500.);
  sprintf(hkey, "maxcol_module_%i",id_);
  meMaxCol_ = theDMBE->book1D(hkey,"Highest Cluster column",500,0.,500.);
  //  sprintf(hkey, "edgehitx_module_%i",id_);
  //  meEdgeHitX_ = theDMBE->book1D(hkey,"X edge hits",500,0.,500.);
  //  sprintf(hkey, "edgehity_module_%i",id_);
  //  meEdgeHitY_ = theDMBE->book1D(hkey,"Y edge hits",500,0.,500.);

  //  sprintf(hkey, "pixclusters_module_%i",id_);
  //mePixClusters_ = theDMBE->book2D(hkey,"Clusters per four pixels",208,0.,416.,80,0.,160.);
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
    //figure out the size of the module/plaquette:
/*    int maxcol=0, maxrow=0;
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      int col = di->column(); // column 
      int row = di->row();    // row
      if(col>maxcol) maxcol=col;
      if(row>maxrow) maxrow=row;
    }*/
    for(di = isearch->data.begin(); di != isearch->data.end(); di++) {
      numberOfClusters++;
      float y = di->y();                   // barycenter y position
      float x = di->x();                   // barycenter x position
      float charge = 0.001*(di->charge()); // total charge of cluster
      int size = di->size();               // total size of cluster (in pixels)
      int sizeX = di->sizeX();             // size of cluster in x-direction
      int sizeY = di->sizeY();             // size of cluster in y-direction
      int minPixelRow = di->minPixelRow(); // min x index
      int maxPixelRow = di->maxPixelRow(); // max x index
      int minPixelCol = di->minPixelCol(); // min y index
      int maxPixelCol = di->maxPixelCol(); // max y index
      //      bool edgeHitX = di->edgeHitX();      // records if a cluster is at the x-edge of the detector
      //      bool edgeHitY = di->edgeHitY();      // records if a cluster is at the y-edge of the detector


	//      (mePixClusters_)->Fill((float)y,(float)x);
      (meY_)->Fill((float)y);
      (meX_)->Fill((float)x);
      (meCharge_)->Fill((float)charge);
      (meSize_)->Fill((int)size);
      (meSizeX_)->Fill((int)sizeX);
      (meSizeY_)->Fill((int)sizeY);
      (meMinRow_)->Fill((int)minPixelRow);
      (meMaxRow_)->Fill((int)maxPixelRow);
      (meMinCol_)->Fill((int)minPixelCol);
      (meMaxCol_)->Fill((int)maxPixelCol);
      //      (meEdgeHitX_)->Fill((int)edgeHitX);
      //      (meEdgeHitY_)->Fill((int)edgeHitY);
            
      /*if(subid==2&&adc>0){
	std::cout<<"Plaquette:"<<side<<" , "<<disk<<" , "<<blade<<" , "
	<<panel<<" , "<<zindex<<" ADC="<<adc<<" , COL="<<col<<" , ROW="<<row<<std::endl;
	}else if(subid==1&&adc>0){
	std::cout<<"Module:"<<layer<<" , "<<ladder<<" , "<<zindex<<" ADC="
	<<adc<<" , COL="<<col<<" , ROW="<<row<<std::endl;
	}*/
    }
    (meNClusters_)->Fill((float)numberOfClusters);
    //std::cout<<"number of clusters="<<numberOfClusters<<std::endl;
    
  }
  
  //std::cout<<"number of detector units="<<numberOfDetUnits<<std::endl;
  
}
