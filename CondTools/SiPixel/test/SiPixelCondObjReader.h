#ifndef CondTools_SiPixel_SiPixelCondObjReader_H
#define CondTools_SiPixel_SiPixelCondObjReader_H
// -*- C++ -*-
//
// Package:    SiPixelCondObjReader
// Class:      SiPixelCondObjReader
// 
/**\class SiPixelCondObjReader SiPixelCondObjReader.h SiPixel/test/SiPixelCondObjReader.h

 Description: Test analyzer for reading pixel calibration from the DB

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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"

namespace cms{
class SiPixelCondObjReader : public edm::EDAnalyzer {

public:

  explicit SiPixelCondObjReader( const edm::ParameterSet& iConfig);

  ~SiPixelCondObjReader(){};
  virtual void beginJob( const edm::EventSetup& );
  virtual void analyze(const edm::Event& , const edm::EventSetup& );
  virtual void endJob() ;

private:

};
}
#endif
