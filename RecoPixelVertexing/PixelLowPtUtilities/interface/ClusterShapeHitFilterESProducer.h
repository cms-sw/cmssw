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
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ClusterShapeHitFilter.h"

//
// class decleration
//

class ClusterShapeHitFilterESProducer : public edm::ESProducer
{
 public:
  ClusterShapeHitFilterESProducer(const edm::ParameterSet&);
  ~ClusterShapeHitFilterESProducer() override;

  typedef std::unique_ptr<ClusterShapeHitFilter> ReturnType;
  ReturnType produce(const ClusterShapeHitFilter::Record &);

 private:
  const std::string pixelShapeFile;
  const std::string pixelShapeFileL1; 
  bool cutOnPixelCharge_, cutOnStripCharge_;
  float minGoodPixelCharge_, minGoodStripCharge_;
  bool cutOnPixelShape_, cutOnStripShape_;
};

#endif
