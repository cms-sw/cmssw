#ifndef SiPixelMonitorCluster_SiPixelClusterModule_h
#define SiPixelMonitorCluster_SiPixelClusterModule_h
// -*- C++ -*-
//
// Package:    SiPixelMonitorDigi
// Class:      SiPixelClusterModule
// 
/*

 Description: Cluster monitoring elements for a Pixel sensor

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo Chiochia & Andrew York
//         Created:  
// $Id: SiPixelClusterModule.h,v 1.4 2007/04/20 21:46:39 andrewdc Exp $
//
//
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/cstdint.hpp>


class SiPixelClusterModule {        

 public:

  /// Default constructor
  SiPixelClusterModule();
  /// Constructor with raw DetId
  SiPixelClusterModule(const uint32_t& id);
  /// Constructor with raw DetId and sensor size
  SiPixelClusterModule(const uint32_t& id, const int& ncols, const int& nrows);
  /// Destructor
  ~SiPixelClusterModule();

  typedef edmNew::DetSet<SiPixelCluster>::const_iterator    ClusterIterator;

  /// Book histograms
  void book(const edm::ParameterSet& iConfig);
  /// Fill histograms
  void fill(const edmNew::DetSetVector<SiPixelCluster> & input);
  
 private:

  uint32_t id_;
  int ncols_;
  int nrows_;
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
  MonitorElement* mePixClusters_;
  //  MonitorElement* meEdgeHitX_;
  //  MonitorElement* meEdgeHitY_;
  
};
#endif
