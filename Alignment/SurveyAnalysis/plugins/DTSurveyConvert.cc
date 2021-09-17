#include <fstream>

#include "Alignment/MuonAlignment/interface/MuonAlignment.h"
#include "Alignment/SurveyAnalysis/interface/DTSurvey.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Alignment/SurveyAnalysis/plugins/DTSurveyConvert.h"

DTSurveyConvert::DTSurveyConvert(const edm::ParameterSet &iConfig) : muonGeoToken_(esConsumes()) {
  //now do what ever initialization is needed
  nameWheel_m2 = iConfig.getUntrackedParameter<std::string>("nameWheel_m2");
  nameWheel_m1 = iConfig.getUntrackedParameter<std::string>("nameWheel_m1");
  nameWheel_0 = iConfig.getUntrackedParameter<std::string>("nameWheel_0");
  nameWheel_p1 = iConfig.getUntrackedParameter<std::string>("nameWheel_p1");
  nameWheel_p2 = iConfig.getUntrackedParameter<std::string>("nameWheel_p2");

  nameChambers_m2 = iConfig.getUntrackedParameter<std::string>("nameChambers_m2");
  nameChambers_m1 = iConfig.getUntrackedParameter<std::string>("nameChambers_m1");
  nameChambers_0 = iConfig.getUntrackedParameter<std::string>("nameChambers_0");
  nameChambers_p1 = iConfig.getUntrackedParameter<std::string>("nameChambers_p1");
  nameChambers_p2 = iConfig.getUntrackedParameter<std::string>("nameChambers_p2");

  wheel_m2 = iConfig.getUntrackedParameter<bool>("wheel_m2");
  wheel_m1 = iConfig.getUntrackedParameter<bool>("wheel_m1");
  wheel_0 = iConfig.getUntrackedParameter<bool>("wheel_0");
  wheel_p1 = iConfig.getUntrackedParameter<bool>("wheel_p1");
  wheel_p2 = iConfig.getUntrackedParameter<bool>("wheel_p2");

  outputFileName = iConfig.getUntrackedParameter<std::string>("OutputTextFile");
  WriteToDB = iConfig.getUntrackedParameter<bool>("writeToDB");
}

// ------------ method called to for each event  ------------
void DTSurveyConvert::analyze(const edm::Event &, const edm::EventSetup &iSetup) {
  const DTGeometry *pDD = &iSetup.getData(muonGeoToken_);

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
  if (wheel_p2 == true) {
    DTSurvey *wheel = new DTSurvey(nameWheel_p2, nameChambers_p2, 2);
    wheel->ReadChambers(pDD);
    wheel->CalculateChambers();
    outFile << *wheel;
    wheelList.push_back(wheel);
  }
  outFile.close();

  if (WriteToDB == true) {
    // Instantiate the helper class
    MuonAlignment align(iSetup);
    std::ifstream inFile(outputFileName.c_str());
    while (!inFile.eof()) {
      float dx, dy, dz, sigma_dx, sigma_dy, sigma_dz;
      float alpha, beta, gamma, sigma_alpha, sigma_beta, sigma_gamma;
      inFile >> dx >> sigma_dx >> dy >> sigma_dy >> dz >> sigma_dz >> alpha >> sigma_alpha >> beta >> sigma_beta >>
          gamma >> sigma_gamma;
      if (inFile.eof())
        break;
      std::vector<float> displacement;
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
