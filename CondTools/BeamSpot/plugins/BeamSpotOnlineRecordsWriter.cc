// -*- C++ -*-
//
// Package:    CondTools/BeamSpot
// Class:      BeamSpotOnlineRecordsWriter
//
/**\class BeamSpotOnlineRecordsWriter BeamSpotOnlineRecordsWriter.cc CondTools/BeamSpot/plugins/BeamSpotOnlineRecordsWriter.cc

 Description: EDAnalyzer to create a BeamSpotOnlineHLTObjectsRcd or BeamSpotOnlineLegacyObjectsRcd payload from a txt file and dump it in a db file

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
#include <ctime>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

//
// class declaration
//

class BeamSpotOnlineRecordsWriter : public edm::one::EDAnalyzer<> {
public:
  explicit BeamSpotOnlineRecordsWriter(const edm::ParameterSet&);
  ~BeamSpotOnlineRecordsWriter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  cond::Time_t pack(uint32_t, uint32_t);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  bool fIsHLT;
  std::ifstream fasciiFile;
  std::string fasciiFileName;
  uint32_t fIOVStartRun;
  uint32_t fIOVStartLumi;
  cond::Time_t fnewSince;
  bool fuseNewSince;
};

//
// constructors and destructor
//
BeamSpotOnlineRecordsWriter::BeamSpotOnlineRecordsWriter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  fIsHLT = iConfig.getParameter<bool>("isHLT");
  fasciiFileName = iConfig.getUntrackedParameter<std::string>("InputFileName");
  fasciiFile.open(fasciiFileName.c_str());
  if (iConfig.exists("IOVStartRun") && iConfig.exists("IOVStartLumi")) {
    fIOVStartRun = iConfig.getUntrackedParameter<uint32_t>("IOVStartRun");
    fIOVStartLumi = iConfig.getUntrackedParameter<uint32_t>("IOVStartLumi");
    fnewSince = BeamSpotOnlineRecordsWriter::pack(fIOVStartRun, fIOVStartLumi);
    fuseNewSince = true;
    edm::LogPrint("BeamSpotOnlineRecordsWriter") << "useNewSince = True";
  } else {
    fuseNewSince = false;
    edm::LogPrint("BeamSpotOnlineRecordsWriter") << "useNewSince = False";
  }
}

BeamSpotOnlineRecordsWriter::~BeamSpotOnlineRecordsWriter() = default;

//
// member functions
//

// ------------ Create a since object (cond::Time_t) by packing Run and LS (both uint32_t)  ------------
cond::Time_t BeamSpotOnlineRecordsWriter::pack(uint32_t fIOVStartRun, uint32_t fIOVStartLumi) {
  return ((uint64_t)fIOVStartRun << 32 | fIOVStartLumi);
}

// ------------ method called for each event  ------------
void BeamSpotOnlineRecordsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {}

// ------------ method called once each job just after ending the event loop  ------------
void BeamSpotOnlineRecordsWriter::endJob() {
  const std::string fLabel = (fIsHLT) ? "BeamSpotOnlineHLTObjectsRcd" : "BeamSpotOnlineLegacyObjectsRcd";
  edm::LogInfo("BeamSpotOnlineRecordsWriter")
      << "Reading " << fLabel << " data from text file: " << fasciiFileName << std::endl;

  // extract from file
  double x, y, z, sigmaZ, dxdz, dydz, beamWidthX, beamWidthY, emittanceX, emittanceY, betastar;
  double cov[7][7];
  int type, lastAnalyzedLumi, firstAnalyzedLumi, lastAnalyzedRun, lastAnalyzedFill;
  std::string tag;
  std::time_t lumiRangeBeginTime, lumiRangeEndTime;

  fasciiFile >> tag >> lastAnalyzedRun;
  fasciiFile >> tag >> tag >> tag >> tag >> lumiRangeBeginTime;  // BeginTimeOfFit parsing (not used in payload)
  fasciiFile >> tag >> tag >> tag >> tag >> lumiRangeEndTime;    // EndTimeOfFit parsing (not used in payload)
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

  // Verify that the parsing was correct by checking the BS positions
  if (std::fabs(x) > 1000. || std::fabs(x) < 1.e-20 || std::fabs(y) > 1000. || std::fabs(y) < 1.e-20 ||
      std::fabs(z) > 1000. || std::fabs(z) < 1.e-20) {
    throw edm::Exception(edm::errors::Unknown)
        << " !!! Error in parsing input file, parsed BS (x,y,z): (" << x << "," << y << "," << z << ") !!!";
  }

  edm::LogPrint("BeamSpotOnlineRecordsWriter") << "---- Parsed these parameters from input txt file ----";
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " lastAnalyzedRun   : " << lastAnalyzedRun;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " lastAnalyzedFill  : " << lastAnalyzedFill;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " firstAnalyzedLumi : " << firstAnalyzedLumi;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " lastAnalyzedLumi  : " << lastAnalyzedLumi;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " lumiRangeBeginTime: " << lumiRangeBeginTime;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " lumiRangeEndTime  : " << lumiRangeEndTime;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " type              : " << type;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " x                 : " << x;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " y                 : " << y;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " z                 : " << z;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " sigmaZ            : " << sigmaZ;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " dxdz              : " << dxdz;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " dydz              : " << dydz;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " beamWidthX        : " << beamWidthX;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " beamWidthY        : " << beamWidthY;
  edm::LogPrint("BeamSpotOnlineRecordsWriter")
      << " Cov(0,j)          : " << cov[0][0] << " " << cov[0][1] << " " << cov[0][2] << " " << cov[0][3] << " "
      << cov[0][4] << " " << cov[0][5] << " " << cov[0][6];
  edm::LogPrint("BeamSpotOnlineRecordsWriter")
      << " Cov(1,j)          : " << cov[1][0] << " " << cov[1][1] << " " << cov[1][2] << " " << cov[1][3] << " "
      << cov[1][4] << " " << cov[1][5] << " " << cov[1][6];
  edm::LogPrint("BeamSpotOnlineRecordsWriter")
      << " Cov(2,j)          : " << cov[2][0] << " " << cov[2][1] << " " << cov[2][2] << " " << cov[2][3] << " "
      << cov[2][4] << " " << cov[2][5] << " " << cov[2][6];
  edm::LogPrint("BeamSpotOnlineRecordsWriter")
      << " Cov(3,j)          : " << cov[3][0] << " " << cov[3][1] << " " << cov[3][2] << " " << cov[3][3] << " "
      << cov[3][4] << " " << cov[3][5] << " " << cov[3][6];
  edm::LogPrint("BeamSpotOnlineRecordsWriter")
      << " Cov(4,j)          : " << cov[4][0] << " " << cov[4][1] << " " << cov[4][2] << " " << cov[4][3] << " "
      << cov[4][4] << " " << cov[4][5] << " " << cov[4][6];
  edm::LogPrint("BeamSpotOnlineRecordsWriter")
      << " Cov(5,j)          : " << cov[5][0] << " " << cov[5][1] << " " << cov[5][2] << " " << cov[5][3] << " "
      << cov[5][4] << " " << cov[5][5] << " " << cov[5][6];
  edm::LogPrint("BeamSpotOnlineRecordsWriter")
      << " Cov(6,j)          : " << cov[6][0] << " " << cov[6][1] << " " << cov[6][2] << " " << cov[6][3] << " "
      << cov[6][4] << " " << cov[6][5] << " " << cov[6][6];
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " emittanceX        : " << emittanceX;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " emittanceY        : " << emittanceY;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " betastar          : " << betastar;
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << "-----------------------------------------------------";

  BeamSpotOnlineObjects abeam;

  abeam.setLastAnalyzedLumi(lastAnalyzedLumi);
  abeam.setLastAnalyzedRun(lastAnalyzedRun);
  abeam.setLastAnalyzedFill(lastAnalyzedFill);
  abeam.setStartTimeStamp(lumiRangeBeginTime);
  abeam.setEndTimeStamp(lumiRangeEndTime);
  abeam.setType(type);
  abeam.setPosition(x, y, z);
  abeam.setSigmaZ(sigmaZ);
  abeam.setdxdz(dxdz);
  abeam.setdydz(dydz);
  abeam.setBeamWidthX(beamWidthX);
  abeam.setBeamWidthY(beamWidthY);
  abeam.setEmittanceX(emittanceX);
  abeam.setEmittanceY(emittanceY);
  abeam.setBetaStar(betastar);

  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 7; ++j) {
      abeam.setCovariance(i, j, cov[i][j]);
    }
  }

  // Set the creation time of the payload to the current time
  auto creationTime =
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  abeam.setCreationTime(creationTime);

  edm::LogPrint("BeamSpotOnlineRecordsWriter") << " Writing results to DB...";

  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable()) {
    edm::LogPrint("BeamSpotOnlineRecordsWriter") << "poolDBService available";
    if (poolDbService->isNewTagRequest(fLabel)) {
      edm::LogPrint("BeamSpotOnlineRecordsWriter") << "new tag requested";
      if (fuseNewSince) {
        edm::LogPrint("BeamSpotOnlineRecordsWriter") << "Using a new Since: " << fnewSince;
        poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, fnewSince, fLabel);
      } else
        poolDbService->createOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->beginOfTime(), fLabel);
    } else {
      edm::LogPrint("BeamSpotOnlineRecordsWriter") << "no new tag requested";
      if (fuseNewSince) {
        edm::LogPrint("BeamSpotOnlineRecordsWriter") << "Using a new Since: " << fnewSince;
        poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, fnewSince, fLabel);
      } else
        poolDbService->appendOneIOV<BeamSpotOnlineObjects>(abeam, poolDbService->currentTime(), fLabel);
    }
  }
  edm::LogPrint("BeamSpotOnlineRecordsWriter") << "[BeamSpotOnlineRecordsWriter] endJob done \n";
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void BeamSpotOnlineRecordsWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("isHLT", true);
  desc.addUntracked<std::string>("InputFileName", "");
  desc.addOptionalUntracked<uint32_t>("IOVStartRun", 1);
  desc.addOptionalUntracked<uint32_t>("IOVStartLumi", 1);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(BeamSpotOnlineRecordsWriter);
