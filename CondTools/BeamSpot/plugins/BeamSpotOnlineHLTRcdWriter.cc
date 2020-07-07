// -*- C++ -*-
//
// Package:    CondTools/BeamSpot
// Class:      BeamSpotOnlineHLTRcdWriter
//
/**\class BeamSpotOnlineHLTRcdWriter BeamSpotOnlineHLTRcdWriter.cc CondTools/BeamSpot/plugins/BeamSpotOnlineHLTRcdWriter.cc

 Description: EDAnalyzer to read the BeamSpotOnlineHLTObjectsRcd and dump it into a txt and root file

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Francesco Brivio
//         Created:  Tue, 11 Feb 2020 11:10:12 GMT
//
//

// system include files
#include <memory>
#include <string>
#include <fstream>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

//
// class declaration
//

class BeamSpotOnlineHLTRcdWriter : public edm::one::EDAnalyzer<> {
public:
  explicit BeamSpotOnlineHLTRcdWriter(const edm::ParameterSet&);
  ~BeamSpotOnlineHLTRcdWriter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  cond::Time_t pack(uint32_t, uint32_t);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  std::ifstream fasciiFile;
  std::string fasciiFileName;
  uint32_t fIOVStartRun;
  uint32_t fIOVStartLumi;
  cond::Time_t fnewSince;
  bool fuseNewSince;

  // ----------member data ---------------------------
};

//
// constructors and destructor
//
BeamSpotOnlineHLTRcdWriter::BeamSpotOnlineHLTRcdWriter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  fasciiFileName = iConfig.getUntrackedParameter<std::string>("InputFileName");
  fasciiFile.open(fasciiFileName.c_str());
  if (iConfig.exists("IOVStartRun") && iConfig.exists("IOVStartLumi")) {
    fIOVStartRun = iConfig.getUntrackedParameter<uint32_t>("IOVStartRun");
    fIOVStartLumi = iConfig.getUntrackedParameter<uint32_t>("IOVStartLumi");
    fnewSince = BeamSpotOnlineHLTRcdWriter::pack(fIOVStartRun, fIOVStartLumi);
    fuseNewSince = true;
  } else {
    fuseNewSince = false;
  }
}

BeamSpotOnlineHLTRcdWriter::~BeamSpotOnlineHLTRcdWriter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ Create a since object (cond::Time_t) by packing Run and LS (both uint32_t)  ------------
cond::Time_t BeamSpotOnlineHLTRcdWriter::pack(uint32_t fIOVStartRun, uint32_t fIOVStartLumi) {
  return ((uint64_t)fIOVStartRun << 32 | fIOVStartLumi);
}

// ------------ method called for each event  ------------
void BeamSpotOnlineHLTRcdWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

// ------------ method called once each job just before starting event loop  ------------
void BeamSpotOnlineHLTRcdWriter::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void BeamSpotOnlineHLTRcdWriter::endJob() {
  std::cout << "Reading BeamSpotOnlineHLTRcd data from text file: " << fasciiFileName << std::endl;

  // extract from file
  double x, y, z, sigmaZ, dxdz, dydz, beamWidthX, beamWidthY, emittanceX, emittanceY, betastar;
  double cov[7][7];
  int type, lastAnalyzedLumi, firstAnalyzedLumi, lastAnalyzedRun, lastAnalyzedFill;
  std::string tag;

  fasciiFile >> tag >> lastAnalyzedRun;
  fasciiFile >> tag >> tag >> tag >> tag >> tag;  // BeginTimeOfFit parsing (not used in payload)
  fasciiFile >> tag >> tag >> tag >> tag >> tag;  // EndTimeOfFit parsing (not used in payload)
  fasciiFile >> tag >> firstAnalyzedLumi;
  fasciiFile >> tag >> lastAnalyzedLumi;
  fasciiFile >> tag >> type;
  fasciiFile >> tag >> x;
  fasciiFile >> tag >> y;
  fasciiFile >> tag >> z;
  fasciiFile >> tag >> sigmaZ;
  fasciiFile >> tag >> dxdz;
  fasciiFile >> tag >> dydz;
  fasciiFile >> tag >> beamWidthX;
  fasciiFile >> tag >> beamWidthY;
  fasciiFile >> tag >> cov[0][0] >> cov[0][1] >> cov[0][2] >> cov[0][3] >> cov[0][4] >> cov[0][5] >> cov[0][6];
  fasciiFile >> tag >> cov[1][0] >> cov[1][1] >> cov[1][2] >> cov[1][3] >> cov[1][4] >> cov[1][5] >> cov[1][6];
  fasciiFile >> tag >> cov[2][0] >> cov[2][1] >> cov[2][2] >> cov[2][3] >> cov[2][4] >> cov[2][5] >> cov[2][6];
  fasciiFile >> tag >> cov[3][0] >> cov[3][1] >> cov[3][2] >> cov[3][3] >> cov[3][4] >> cov[3][5] >> cov[3][6];
  fasciiFile >> tag >> cov[4][0] >> cov[4][1] >> cov[4][2] >> cov[4][3] >> cov[4][4] >> cov[4][5] >> cov[4][6];
  fasciiFile >> tag >> cov[5][0] >> cov[5][1] >> cov[5][2] >> cov[5][3] >> cov[5][4] >> cov[5][5] >> cov[5][6];
  fasciiFile >> tag >> cov[6][0] >> cov[6][1] >> cov[6][2] >> cov[6][3] >> cov[6][4] >> cov[6][5] >> cov[6][6];
  fasciiFile >> tag >> emittanceX;
  fasciiFile >> tag >> emittanceY;
  fasciiFile >> tag >> betastar;

  lastAnalyzedFill = -999;

  std::cout << "---- Parsed these parameters from input txt file ----" << std::endl;
  std::cout << " lastAnalyzedRun   : " << lastAnalyzedRun << std::endl;
  std::cout << " lastAnalyzedFill  : " << lastAnalyzedFill << std::endl;
  std::cout << " firstAnalyzedLumi : " << firstAnalyzedLumi << std::endl;
  std::cout << " lastAnalyzedLumi  : " << lastAnalyzedLumi << std::endl;
  std::cout << " type              : " << type << std::endl;
  std::cout << " x                 : " << x << std::endl;
  std::cout << " y                 : " << y << std::endl;
  std::cout << " z                 : " << z << std::endl;
  std::cout << " sigmaZ            : " << sigmaZ << std::endl;
  std::cout << " dxdz              : " << dxdz << std::endl;
  std::cout << " dydz              : " << dydz << std::endl;
  std::cout << " beamWidthX        : " << beamWidthX << std::endl;
  std::cout << " beamWidthY        : " << beamWidthY << std::endl;
  std::cout << " Cov(0,j)          : " << cov[0][0] << " " << cov[0][1] << " " << cov[0][2] << " " << cov[0][3] << " "
            << cov[0][4] << " " << cov[0][5] << " " << cov[0][6] << std::endl;
  std::cout << " Cov(1,j)          : " << cov[1][0] << " " << cov[1][1] << " " << cov[1][2] << " " << cov[1][3] << " "
            << cov[1][4] << " " << cov[1][5] << " " << cov[1][6] << std::endl;
  std::cout << " Cov(2,j)          : " << cov[2][0] << " " << cov[2][1] << " " << cov[2][2] << " " << cov[2][3] << " "
            << cov[2][4] << " " << cov[2][5] << " " << cov[2][6] << std::endl;
  std::cout << " Cov(3,j)          : " << cov[3][0] << " " << cov[3][1] << " " << cov[3][2] << " " << cov[3][3] << " "
            << cov[3][4] << " " << cov[3][5] << " " << cov[3][6] << std::endl;
  std::cout << " Cov(4,j)          : " << cov[4][0] << " " << cov[4][1] << " " << cov[4][2] << " " << cov[4][3] << " "
            << cov[4][4] << " " << cov[4][5] << " " << cov[4][6] << std::endl;
  std::cout << " Cov(5,j)          : " << cov[5][0] << " " << cov[5][1] << " " << cov[5][2] << " " << cov[5][3] << " "
            << cov[5][4] << " " << cov[5][5] << " " << cov[5][6] << std::endl;
  std::cout << " Cov(6,j)          : " << cov[6][0] << " " << cov[6][1] << " " << cov[6][2] << " " << cov[6][3] << " "
            << cov[6][4] << " " << cov[6][5] << " " << cov[6][6] << std::endl;
  std::cout << " emittanceX        : " << emittanceX << std::endl;
  std::cout << " emittanceY        : " << emittanceY << std::endl;
  std::cout << " betastar          : " << betastar << std::endl;
  std::cout << "-----------------------------------------------------" << std::endl;

  BeamSpotOnlineObjects* abeam = new BeamSpotOnlineObjects();

  abeam->SetLastAnalyzedLumi(lastAnalyzedLumi);
  abeam->SetLastAnalyzedRun(lastAnalyzedRun);
  abeam->SetLastAnalyzedFill(lastAnalyzedFill);
  abeam->SetType(type);
  abeam->SetPosition(x, y, z);
  abeam->SetSigmaZ(sigmaZ);
  abeam->Setdxdz(dxdz);
  abeam->Setdydz(dydz);
  abeam->SetBeamWidthX(beamWidthX);
  abeam->SetBeamWidthY(beamWidthY);
  abeam->SetEmittanceX(emittanceX);
  abeam->SetEmittanceY(emittanceY);
  abeam->SetBetaStar(betastar);

  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      abeam->SetCovariance(i, j, cov[i][j]);
    }
  }

  std::cout << " Writing results to DB..." << std::endl;

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    std::cout << "poolDBService available" << std::endl;
    if (poolDbService->isNewTagRequest("BeamSpotOnlineHLTObjectsRcd")) {
      std::cout << "new tag requested" << std::endl;
      if (fuseNewSince) {
        std::cout << "Using a new Since: " << fnewSince << std::endl;
        poolDbService->createNewIOV<BeamSpotOnlineObjects>(
            abeam, fnewSince, poolDbService->endOfTime(), "BeamSpotOnlineHLTObjectsRcd");
      } else
        poolDbService->createNewIOV<BeamSpotOnlineObjects>(
            abeam, poolDbService->beginOfTime(), poolDbService->endOfTime(), "BeamSpotOnlineHLTObjectsRcd");
    } else {
      std::cout << "no new tag requested" << std::endl;
      if (fuseNewSince) {
        std::cout << "Using a new Since: " << fnewSince << std::endl;
        poolDbService->appendSinceTime<BeamSpotOnlineObjects>(abeam, fnewSince, "BeamSpotOnlineHLTObjectsRcd");
      } else
        poolDbService->appendSinceTime<BeamSpotOnlineObjects>(
            abeam, poolDbService->currentTime(), "BeamSpotOnlineHLTObjectsRcd");
    }
  }

  std::cout << "[BeamSpotOnlineHLTRcdWriter] endJob done \n" << std::endl;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamSpotOnlineHLTRcdWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotOnlineHLTRcdWriter);
