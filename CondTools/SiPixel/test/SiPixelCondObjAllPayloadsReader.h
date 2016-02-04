#ifndef CondTools_SiPixel_SiPixelCondObjAllPayloadsReader_H
#define CondTools_SiPixel_SiPixelCondObjAllPayloadsReader_H
// -*- C++ -*-
//
// Package:    SiPixelCondObjAllPayloadsReader
// Class:      SiPixelCondObjAllPayloadsReader
// 
/**\class SiPixelCondObjAllPayloadsReader SiPixelCondObjAllPayloadsReader.h SiPixel/test/SiPixelCondObjAllPayloadsReader.h

 Description: Test analyzer for reading pixel calibration from the DB

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vincenzo CHIOCHIA
//         Created:  Tue Oct 17 17:40:56 CEST 2006
// $Id: SiPixelCondObjAllPayloadsReader.h,v 1.5 2010/01/12 11:29:54 rougny Exp $
//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationServiceBase.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationOfflineService.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationForHLTService.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include <string>

namespace cms{
class SiPixelCondObjAllPayloadsReader : public edm::EDAnalyzer {

public:

  explicit SiPixelCondObjAllPayloadsReader( const edm::ParameterSet& iConfig);

  ~SiPixelCondObjAllPayloadsReader(){};
  virtual void beginJob();
  virtual void analyze(const edm::Event& , const edm::EventSetup& );
  virtual void endJob() ;

private:

  edm::ParameterSet conf_;
  edm::ESHandle<TrackerGeometry> tkgeom;
  //edm::ESHandle<SiPixelGainCalibration> SiPixelGainCalibration_;
  SiPixelGainCalibrationServiceBase* SiPixelGainCalibrationService_;

  std::map< uint32_t, TH1F* >  _TH1F_Pedestals_m;
  std::map< uint32_t, TH1F* >  _TH1F_Gains_m;
  TH1F *_TH1F_Gains_sum;
  TH1F *_TH1F_Pedestals_sum;
  TH1F *_TH1F_Gains_all;
  TH1F *_TH1F_Pedestals_all;

};
}
#endif
