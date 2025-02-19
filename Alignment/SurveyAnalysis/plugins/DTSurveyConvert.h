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
// $Id: DTSurveyConvert.h,v 1.1 2008/04/11 05:06:12 cklae Exp $
//
//

#include "FWCore/Framework/interface/EDAnalyzer.h"

class DTSurvey;

class DTSurveyConvert : public edm::EDAnalyzer
{
   public:
      explicit DTSurveyConvert(const edm::ParameterSet&);

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob(const edm::EventSetup&);
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

