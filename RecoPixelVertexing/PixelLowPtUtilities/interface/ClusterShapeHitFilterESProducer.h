#ifndef RecoPixelVertexing_PixelLowPtUtilities_ClusterShapeHitFilterESProducer_H
#define RecoPixelVertexing_PixelLowPtUtilities_ClusterShapeHitFilterESProducer_H


// -*- C++ -*-
//
// Package:    ClusterShapeHitFilterESProducer
// Class:      ClusterShapeHitFilterESProducer
// 
/**\class ClusterShapeHitFilterESProducer ClusterShapeHitFilterESProducer.h
 * TrackingTools/ClusterShapeHitFilterESProducer/src/ClusterShapeHitFilterESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri Sep 28 18:07:52 CEST 2007
// $Id: ClusterShapeHitFilterESProducer.h,v 1.2 2009/02/26 15:07:47 sikler Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

//
// class decleration
//

class ClusterShapeHitFilterESProducer : public edm::ESProducer
{
 public:
  ClusterShapeHitFilterESProducer(const edm::ParameterSet&);
  ~ClusterShapeHitFilterESProducer();

//  typedef std::auto_ptr<TrajectoryFilter> ReturnType;
//  ReturnType produce(const TrajectoryFilter::Record &);
  typedef std::auto_ptr<ClusterShapeHitFilter> ReturnType;
  ReturnType produce(const TrajectoryFilter::Record &);

 private:
  std::string componentName;
  std::string componentType;
  edm::ParameterSet filterPset;
};

#endif
