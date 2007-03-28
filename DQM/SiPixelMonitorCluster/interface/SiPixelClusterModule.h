#ifndef SiPixelMonitorCluster_SiPixelClusterModule_h
#define SiPixelMonitorCluster_SiPixelClusterModule_h

#include "DQMServices/Core/interface/MonitorElement.h"

//#include "DataFormats/SiPixelCluster/interface/PixelClusterCollection.h"
//#include "DataFormats/SiPixelCluster/interface/PixelCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <boost/cstdint.hpp>

class SiPixelClusterModule {        

 public:
  
  SiPixelClusterModule();
  
  SiPixelClusterModule(uint32_t id);
  
  ~SiPixelClusterModule();

  typedef edm::DetSet<SiPixelCluster>::const_iterator    ClusterIterator;

  void book();

  //void fill(const PixelClusterCollection* clusterCollection);
  void fill(const edm::DetSetVector<SiPixelCluster> & input);
  
 private:
  uint32_t id_;
  MonitorElement* meNClusters_;
  MonitorElement* meY_;
  MonitorElement* meX_;
  MonitorElement* meCharge_;
  MonitorElement* meSize_;
  MonitorElement* meSizeX_;
  MonitorElement* meSizeY_;
  MonitorElement* meMinRow_;
  MonitorElement* meMaxRow_;
  MonitorElement* meMinCol_;
  MonitorElement* meMaxCol_;
  //  MonitorElement* meEdgeHitX_;
  //  MonitorElement* meEdgeHitY_;
  //  MonitorElement* mePixClusters_;
  
};
#endif
