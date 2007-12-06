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
// $Id: DTSurveyConvert.cpp,v 1.4 2007/04/27 17:05:31 pablom Exp $
//
//


// system include files
#include <memory>

#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Alignment/MuonAlignment/interface/MuonAlignment.h"

#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "Alignment/SurveyAnalysis/interface/DTSurvey.h"


//
// class decleration
//

class DTSurveyConvert : public edm::EDAnalyzer {
   public:
      explicit DTSurveyConvert(const edm::ParameterSet&);
      ~DTSurveyConvert();
      

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob(const edm::EventSetup&);
      std::vector<DTSurvey *> wheelList;
      string nameWheel_m2; 
      string nameWheel_m1; 
      string nameWheel_0; 
      string nameWheel_p1; 
      string nameWheel_p2;
      string nameChambers_m2;
      string nameChambers_m1;
      string nameChambers_0;
      string nameChambers_p1;
      string nameChambers_p2;
      string outputFileName;
      bool wheel_m2;
      bool wheel_m1;
      bool wheel_0;
      bool wheel_p1;
      bool wheel_p2;
      bool WriteToDB;

 

      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DTSurveyConvert::DTSurveyConvert(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
   nameWheel_m2 = iConfig.getUntrackedParameter<string>("nameWheel_m2");
   nameWheel_m1 = iConfig.getUntrackedParameter<string>("nameWheel_m1");
   nameWheel_0 = iConfig.getUntrackedParameter<string>("nameWheel_0");
   nameWheel_p1 = iConfig.getUntrackedParameter<string>("nameWheel_p1");
   nameWheel_p2 = iConfig.getUntrackedParameter<string>("nameWheel_p2");
   
   nameChambers_m2 = iConfig.getUntrackedParameter<string>("nameChambers_m2");
   nameChambers_m1 = iConfig.getUntrackedParameter<string>("nameChambers_m1");
   nameChambers_0 = iConfig.getUntrackedParameter<string>("nameChambers_0");
   nameChambers_p1 = iConfig.getUntrackedParameter<string>("nameChambers_p1");
   nameChambers_p2 = iConfig.getUntrackedParameter<string>("nameChambers_p2");

   wheel_m2 = iConfig.getUntrackedParameter<bool>("wheel_m2");
   wheel_m1 = iConfig.getUntrackedParameter<bool>("wheel_m1");
   wheel_0 = iConfig.getUntrackedParameter<bool>("wheel_0");
   wheel_p1 = iConfig.getUntrackedParameter<bool>("wheel_p1");
   wheel_p2 = iConfig.getUntrackedParameter<bool>("wheel_p2");

   outputFileName = iConfig.getUntrackedParameter<string>("OutputTextFile");
   WriteToDB = iConfig.getUntrackedParameter<bool>("writeToDB");
  
}


DTSurveyConvert::~DTSurveyConvert()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DTSurveyConvert::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::ESHandle<DTGeometry> pDD;
  iSetup.get<MuonGeometryRecord>().get( pDD );
 
  std::ofstream outFile(outputFileName.c_str());
  
  if (wheel_m2 == true) {
    DTSurvey *wheel = new DTSurvey(nameWheel_m2, nameChambers_m2, -2);
    wheel->ReadChambers(pDD);
    wheel->CalculateChambers();
    outFile << *wheel; 
    wheelList.push_back(wheel);
  } 
  if (wheel_m1 == true) {
    DTSurvey *wheel = new DTSurvey(nameWheel_m1, nameChambers_m1, -1);
    wheel->ReadChambers(pDD);
    wheel->CalculateChambers();
    outFile << *wheel; 
    wheelList.push_back(wheel);
  } 
  if (wheel_0 == true) {
    DTSurvey *wheel = new DTSurvey(nameWheel_0, nameChambers_0, 0);
    wheel->ReadChambers(pDD);
    wheel->CalculateChambers();
    outFile << *wheel; 
    wheelList.push_back(wheel);
  } 
  if (wheel_p1 == true) {
    DTSurvey *wheel = new DTSurvey(nameWheel_p1, nameChambers_p1, 1);
    wheel->ReadChambers(pDD);
    wheel->CalculateChambers();
    outFile << *wheel; 
    wheelList.push_back(wheel);
  } 
  if(wheel_p2 == true) {
    DTSurvey *wheel = new DTSurvey(nameWheel_p2, nameChambers_p2, 2);
    wheel->ReadChambers(pDD);
    wheel->CalculateChambers();
    outFile << *wheel; 
    wheelList.push_back(wheel);
  }
  outFile.close();
}


// ------------ method called once each job just before starting event loop  ------------
void 
DTSurveyConvert::beginJob(const edm::EventSetup& eventSetup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DTSurveyConvert::endJob(const edm::EventSetup& eventSetup) {
  if(WriteToDB == true) {
    // Instantiate the helper class
    MuonAlignment align( eventSetup );
    std::ifstream inFile(outputFileName.c_str());
    while(!inFile.eof()) {
      float dx, dy, dz, sigma_dx, sigma_dy, sigma_dz;
      float alpha, beta, gamma, sigma_alpha, sigma_beta, sigma_gamma;
      inFile >> dx >> sigma_dx >> dy >> sigma_dy >> dz >> sigma_dz
             >> alpha >> sigma_alpha >> beta >> sigma_beta >> gamma >> sigma_gamma; 
      if(inFile.eof()) break;
      vector<float> displacement;
      vector<float> rotation;
      displacement.push_back(dx);
      displacement.push_back(dy);
      displacement.push_back(dz);
      displacement.push_back(-alpha);
      displacement.push_back(-beta);
      displacement.push_back(-gamma);
    }
    inFile.close();
    align.saveToDB();
  }
}

DEFINE_FWK_MODULE(DTSurveyConvert);

#endif

