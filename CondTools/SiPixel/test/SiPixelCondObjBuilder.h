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
// $Id: SiPixelCondObjBuilder.h,v 1.10 2010/01/12 11:29:54 rougny Exp $
//
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibration.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelGainCalibrationService.h"
#include "CondFormats/SiPixelObjects/interface/PixelIndices.h"
#include <string>

namespace cms{
class SiPixelCondObjBuilder : public edm::EDAnalyzer {

public:

  explicit SiPixelCondObjBuilder( const edm::ParameterSet& iConfig);

  ~SiPixelCondObjBuilder(){};
  virtual void beginJob();
  virtual void analyze(const edm::Event& , const edm::EventSetup& );
  virtual void endJob() ;
  bool loadFromFile();

private:

  edm::ParameterSet conf_;
  bool appendMode_;
  SiPixelGainCalibration* SiPixelGainCalibration_;
  SiPixelGainCalibrationService SiPixelGainCalibrationService_;
  std::string recordName_;

  double meanPed_;
  double rmsPed_;
  double meanGain_;
  double rmsGain_;
  double secondRocRowGainOffset_;
  double secondRocRowPedOffset_;
  int numberOfModules_;
  bool fromFile_;
  std::string fileName_; 

  // Internal class
  class CalParameters {
  public:
    float p0;
    float p1;
  };
  // Map for storing calibration constants
  std::map<int,CalParameters, std::less<int> > calmap_;
  PixelIndices * pIndexConverter_; // Pointer to the index converter 

};
}
#endif
