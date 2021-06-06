
/*
 *  See header file for a description of this class.
 *
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */

#include "Calibration/TkAlCaRecoProducers/interface/AlcaBeamSpotHarvester.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "CondCore/Utilities/bin/cmscond_export_iov.cpp"
//#include "CondCore/Utilities/interface/Utilities.h"
// #include "FWCore/MessageLogger/interface/JobReport.h"

#include <cstring>
#include <iostream>

using namespace edm;
using namespace reco;
// using namespace std;

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotHarvester::AlcaBeamSpotHarvester(const edm::ParameterSet &iConfig)
    : beamSpotOutputBase_(iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters")
                              .getUntrackedParameter<std::string>("BeamSpotOutputBase")),
      outputrecordName_(iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters")
                            .getUntrackedParameter<std::string>("outputRecordName", "BeamSpotObjectsRcd")),
      sigmaZValue_(iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters")
                       .getUntrackedParameter<double>("SigmaZValue")),
      sigmaZCut_(iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters")
                     .getUntrackedParameter<double>("SigmaZCut")),
      dumpTxt_(
          iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<bool>("DumpTxt")),
      outTxtFileName_(iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters")
                          .getUntrackedParameter<std::string>("TxtFileName")),
      theAlcaBeamSpotManager_(iConfig, consumesCollector()) {}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::analyze(const edm::Event &iEvent, const edm::EventSetup &) {
  //  edm::LogInfo("AlcaBeamSpotHarvester")
  //      << "Lumi: " << iEvent.luminosityBlock()
  //      << " Time: " << iEvent.time().unixTime()
  //      << std::endl;
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::beginRun(const edm::Run &, const edm::EventSetup &) { theAlcaBeamSpotManager_.reset(); }

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::endRun(const edm::Run &iRun, const edm::EventSetup &) {
  theAlcaBeamSpotManager_.createWeightedPayloads();
  std::map<edm::LuminosityBlockNumber_t, std::pair<edm::Timestamp, reco::BeamSpot>> beamSpotMap =
      theAlcaBeamSpotManager_.getPayloads();
  Service<cond::service::PoolDBOutputService> poolDbService;
  //  cond::ExportIOVUtilities utilities;

  std::string outTxt = Form("%s_Run%d.txt", outTxtFileName_.c_str(), iRun.id().run());
  std::ofstream outFile;
  outFile.open(outTxt.c_str(), std::ios::app);

  if (poolDbService.isAvailable()) {
    for (AlcaBeamSpotManager::bsMap_iterator it = beamSpotMap.begin(); it != beamSpotMap.end(); it++) {
      BeamSpotObjects *aBeamSpot = new BeamSpotObjects();
      aBeamSpot->SetType(it->second.second.type());
      aBeamSpot->SetPosition(it->second.second.x0(), it->second.second.y0(), it->second.second.z0());
      if (sigmaZValue_ == -1) {
        aBeamSpot->SetSigmaZ(it->second.second.sigmaZ());
      } else {
        aBeamSpot->SetSigmaZ(sigmaZValue_);
      }
      aBeamSpot->Setdxdz(it->second.second.dxdz());
      aBeamSpot->Setdydz(it->second.second.dydz());
      aBeamSpot->SetBeamWidthX(it->second.second.BeamWidthX());
      aBeamSpot->SetBeamWidthY(it->second.second.BeamWidthY());
      aBeamSpot->SetEmittanceX(it->second.second.emittanceX());
      aBeamSpot->SetEmittanceY(it->second.second.emittanceY());
      aBeamSpot->SetBetaStar(it->second.second.betaStar());

      for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 7; ++j) {
          aBeamSpot->SetCovariance(i, j, it->second.second.covariance(i, j));
        }
      }

      if (sigmaZValue_ > 0) {
        aBeamSpot->SetCovariance(3, 3, 0.000025);
      }

      cond::Time_t thisIOV = 1;

      beamspot::BeamSpotContainer currentBS;

      // run based
      if (beamSpotOutputBase_ == "runbased") {
        thisIOV = (cond::Time_t)iRun.id().run();
      }
      // lumi based
      else if (beamSpotOutputBase_ == "lumibased") {
        edm::LuminosityBlockID lu(iRun.id().run(), it->first);
        thisIOV = (cond::Time_t)(lu.value());

        currentBS.beamspot = it->second.second;
        currentBS.run = iRun.id().run();
        currentBS.beginLumiOfFit = it->first;
        currentBS.endLumiOfFit = it->first;  // endLumi = initLumi

        std::time_t lumi_t_begin = it->second.first.unixTime();
        std::time_t lumi_t_end = it->second.first.unixTime();  // begin time == end time
        strftime(
            currentBS.beginTimeOfFit, sizeof currentBS.beginTimeOfFit, "%Y.%m.%d %H:%M:%S GMT", gmtime(&lumi_t_begin));
        strftime(currentBS.endTimeOfFit, sizeof currentBS.endTimeOfFit, "%Y.%m.%d %H:%M:%S GMT", gmtime(&lumi_t_end));

        currentBS.reftime[0] = lumi_t_begin;
        currentBS.reftime[1] = lumi_t_end;
      }
      if (poolDbService->isNewTagRequest(outputrecordName_)) {
        edm::LogInfo("AlcaBeamSpotHarvester") << "new tag requested" << std::endl;
        // poolDbService->createNewIOV<BeamSpotObjects>(aBeamSpot,
        // poolDbService->beginOfTime(),poolDbService->endOfTime(),"BeamSpotObjectsRcd");
        // poolDbService->createNewIOV<BeamSpotObjects>(aBeamSpot,
        // poolDbService->currentTime(),
        // poolDbService->endOfTime(),"BeamSpotObjectsRcd");
        poolDbService->writeOne<BeamSpotObjects>(aBeamSpot, thisIOV, outputrecordName_);
        if (dumpTxt_ && beamSpotOutputBase_ == "lumibased") {
          beamspot::dumpBeamSpotTxt(outFile, currentBS);

          edm::Service<edm::JobReport> jr;
          if (jr.isAvailable()) {
            std::map<std::string, std::string> jrInfo;
            jrInfo["Source"] = std::string("AlcaHarvesting");
            jrInfo["FileClass"] = std::string("ALCATXT");
            jr->reportAnalysisFile(outTxt, jrInfo);
          }
        }
      } else {
        edm::LogInfo("AlcaBeamSpotHarvester") << "no new tag requested, appending IOV" << std::endl;
        // poolDbService->appendSinceTime<BeamSpotObjects>(aBeamSpot,
        // poolDbService->currentTime(),"BeamSpotObjectsRcd");
        poolDbService->writeOne<BeamSpotObjects>(aBeamSpot, thisIOV, outputrecordName_);
        if (dumpTxt_ && beamSpotOutputBase_ == "lumibased") {
          beamspot::dumpBeamSpotTxt(outFile, currentBS);
        }
      }

      /*
            int         argc = 15;
            const char* argv[] = {"endRun"
                                 ,"-d","sqlite_file:combined.db"
                                 ,"-s","sqlite_file:testbs2.db"
                                 ,"-l","sqlite_file:log.db"
                                 ,"-i","TestLSBasedBS"
                                 ,"-t","TestLSBasedBS"
                                 ,"-b","1"
                                 ,"-e","10"
                                 };

            edm::LogInfo("AlcaBeamSpotHarvester")
              << "Running utilities!"
              << utilities.run(argc,(char**)argv);
            edm::LogInfo("AlcaBeamSpotHarvester")
              << "Run utilities!"
              << std::endl;
      */
    }
  }

  outFile.close();
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) {}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::endLuminosityBlock(const edm::LuminosityBlock &iLumi, const edm::EventSetup &) {
  theAlcaBeamSpotManager_.readLumi(iLumi);
}

DEFINE_FWK_MODULE(AlcaBeamSpotHarvester);
