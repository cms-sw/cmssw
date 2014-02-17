#ifndef CalibTracker_SiPixelESProducers_test_SiPixelFakeGainForHLTReader
#define CalibTracker_SiPixelESProducers_test_SiPixelFakeGainForHLTReader
// -*- C++ -*-
//
// Package:    SiPixelFakeGainForHLTReader
// Class:      SiPixelFakeGainForHLTReader
// 
/**\class SiPixelFakeGainForHLTReader SiPixelFakeGainForHLTReader.h SiPixelESProducers/test/SiPixelFakeGainForHLTReader.h

 Description: Test analyzer for fake pixel calibrationForHLT

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
// $Id: SiPixelFakeGainForHLTReader.h,v 1.3 2010/01/13 16:25:39 ursl Exp $
//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"

namespace cms{
class SiPixelFakeGainForHLTReader : public edm::EDAnalyzer {

public:

  explicit SiPixelFakeGainForHLTReader( const edm::ParameterSet& iConfig);

  ~SiPixelFakeGainForHLTReader(){};
  virtual void beginJob() {;}
  virtual void beginRun(const edm::Run& , const edm::EventSetup& );
  virtual void analyze(const edm::Event& , const edm::EventSetup& );
  virtual void endJob() ;

private:

  edm::ParameterSet conf_;
  edm::ESHandle<TrackerGeometry> tkgeom;
  //edm::ESHandle<SiPixelGainCalibrationForHLT> SiPixelGainCalibrationForHLT_;
  SiPixelGainCalibrationForHLTService SiPixelGainCalibrationForHLTService_;

  std::map< uint32_t, TH1F* >  _TH1F_Pedestals_m;
  std::map< uint32_t, TH1F* >  _TH1F_Gains_m;
  std::string filename_;
  TFile* fFile;

};
}
#endif
