#ifndef Alignment_SurveyAnalysis_DTSurveyConvert_H
#define Alignment_SurveyAnalysis_DTSurveyConvert_H

// -*- C++ -*-
//
// Package:    DTSurveyConvert
// Class:      DTSurveyConvert
//
/**\class DTSurveyConvert DTSurveyConvert.cc Alignment/DTSurveyConvert/src/DTSurveyConvert.cc

 Description: Reads survey information, process it and outputs a text file with results

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pablo Martinez Ruiz Del Arbol
//         Created:  Wed Mar 28 09:50:08 CEST 2007
//
//

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class DTSurvey;

class DTSurveyConvert : public edm::one::EDAnalyzer<> {
public:
  explicit DTSurveyConvert(const edm::ParameterSet &);

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> muonGeoToken_;

  std::vector<DTSurvey *> wheelList;
  std::string nameWheel_m2;
  std::string nameWheel_m1;
  std::string nameWheel_0;
  std::string nameWheel_p1;
  std::string nameWheel_p2;
  std::string nameChambers_m2;
  std::string nameChambers_m1;
  std::string nameChambers_0;
  std::string nameChambers_p1;
  std::string nameChambers_p2;
  std::string outputFileName;
  bool wheel_m2;
  bool wheel_m1;
  bool wheel_0;
  bool wheel_p1;
  bool wheel_p2;
  bool WriteToDB;
};

#endif
