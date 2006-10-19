#ifndef CondTools_SiPixel_SiPixelCondObjBuilder_H
#define CondTools_SiPixel_SiPixelCondObjBuilder_H
// -*- C++ -*-
//
// Package:    SiPixelCondObjBuilder
// Class:      SiPixelCondObjBuilder
// 
/**\class SiPixelCondObjBuilder SiPixelCondObjBuilder.h SiPixel/test/SiPixelCondObjBuilder.h

 Description: Test analyzer for writing pixel calibration in the DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
// $Id$
//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
namespace cms{
class SiPixelCondObjBuilder : public edm::EDAnalyzer {

public:

  explicit SiPixelCondObjBuilder( const edm::ParameterSet& iConfig);

  ~SiPixelCondObjBuilder(){};
  virtual void beginJob( const edm::EventSetup& );
  virtual void analyze(const edm::Event& , const edm::EventSetup& );
  virtual void endJob() ;

private:
  bool appendMode_;
  SiPixelGainCalibration* SiPixelGainCalibration_;
};
}
#endif
