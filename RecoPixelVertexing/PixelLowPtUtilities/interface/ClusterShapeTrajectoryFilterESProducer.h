#ifndef RecoPixelVertexing_PixelLowPtUtilities_ClusterShapeTrajectoryFilterESProducer_H
#define RecoPixelVertexing_PixelLowPtUtilities_ClusterShapeTrajectoryFilterESProducer_H


// -*- C++ -*-
//
// Package:    ClusterShapeTrajectoryFilterESProducer
// Class:      ClusterShapeTrajectoryFilterESProducer
// 
/**\class ClusterShapeTrajectoryFilterESProducer ClusterShapeTrajectoryFilterESProducer.h TrackingTools/ClusterShapeTrajectoryFilterESProducer/src/ClusterShapeTrajectoryFilterESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri Sep 28 18:07:52 CEST 2007
// $Id: ClusterShapeTrajectoryFilterESProducer.h,v 1.2 2009/02/26 15:07:47 sikler Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

//#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeTrajectoryFilter.h"

//class TrajectoryFilter;
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"

//
// class decleration
//

class ClusterShapeTrajectoryFilterESProducer : public edm::ESProducer
{
 public:
  ClusterShapeTrajectoryFilterESProducer(const edm::ParameterSet&);
  ~ClusterShapeTrajectoryFilterESProducer();

  typedef std::auto_ptr<TrajectoryFilter> ReturnType;

  ReturnType produce(const TrajectoryFilter::Record &);

 private:
  std::string componentName;
  std::string componentType;
  edm::ParameterSet filterPset;
};

#endif
