
/*
 *  See header file for a description of this class.
 *
 *  \author L. Uplegger F. Yumiceva - Fermilab
 */

#include "Calibration/TkAlCaRecoProducers/interface/AlcaBeamSpotHarvester.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/FileBlock.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
//#include "CondCore/Utilities/bin/cmscond_export_iov.cpp"
//#include "CondCore/Utilities/interface/Utilities.h"
// #include "FWCore/MessageLogger/interface/JobReport.h"

#include <iostream> 

using namespace edm;
using namespace reco;
//using namespace std;

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotHarvester::AlcaBeamSpotHarvester(const edm::ParameterSet& iConfig) :
  beamSpotOutputBase_    (iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<std::string>("BeamSpotOutputBase")),
  outputrecordName_      (iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<std::string>("outputRecordName", "BeamSpotObjectsRcd")),
  sigmaZValue_           (iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<double>("SigmaZValue")),
  sigmaZCut_           (iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<double>("SigmaZCut")),
  dumpTxt_               (iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<bool>("DumpTxt")),
  outTxtFileName_        (iConfig.getParameter<ParameterSet>("AlcaBeamSpotHarvesterParameters").getUntrackedParameter<std::string>("TxtFileName")),
  theAlcaBeamSpotManager_(iConfig, consumesCollector()) {  
}

//--------------------------------------------------------------------------------------------------
AlcaBeamSpotHarvester::~AlcaBeamSpotHarvester(){}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::beginJob() {}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::endJob() {}  

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
//  edm::LogInfo("AlcaBeamSpotHarvester")
//      << "Lumi: " << iEvent.luminosityBlock() 
//      << " Time: " << iEvent.time().unixTime() 
//      << std::endl;
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::beginRun(const edm::Run&, const edm::EventSetup&){
  theAlcaBeamSpotManager_.reset();
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::endRun(const edm::Run& iRun, const edm::EventSetup&){
  theAlcaBeamSpotManager_.createWeightedPayloads();
  std::map<edm::LuminosityBlockNumber_t,reco::BeamSpot> beamSpotMap = theAlcaBeamSpotManager_.getPayloads();
  Service<cond::service::PoolDBOutputService> poolDbService;
//  cond::ExportIOVUtilities utilities;

  std::string outTxt = Form("%s_Run%d.txt", outTxtFileName_.c_str(), iRun.id().run());

  if(poolDbService.isAvailable() ) {
    for(AlcaBeamSpotManager::bsMap_iterator it=beamSpotMap.begin(); it!=beamSpotMap.end();it++){
      BeamSpotObjects *aBeamSpot = new BeamSpotObjects();
      aBeamSpot->SetType(it->second.type());
      aBeamSpot->SetPosition(it->second.x0(),it->second.y0(),it->second.z0());
      if(sigmaZValue_ == -1){
        aBeamSpot->SetSigmaZ(it->second.sigmaZ());
      }
      else{
        aBeamSpot->SetSigmaZ(sigmaZValue_);
      }
      aBeamSpot->Setdxdz(it->second.dxdz());
      aBeamSpot->Setdydz(it->second.dydz());
      aBeamSpot->SetBeamWidthX(it->second.BeamWidthX());
      aBeamSpot->SetBeamWidthY(it->second.BeamWidthY());
      aBeamSpot->SetEmittanceX(it->second.emittanceX());
      aBeamSpot->SetEmittanceY(it->second.emittanceY());
      aBeamSpot->SetBetaStar(it->second.betaStar() );
	
      for (int i=0; i<7; ++i) {
	for (int j=0; j<7; ++j) {
	  aBeamSpot->SetCovariance(i,j,it->second.covariance(i,j));
	}
      }

      if(sigmaZValue_ > 0){
        aBeamSpot->SetCovariance(3,3,0.000025);
      }

      cond::Time_t thisIOV = 1;

      beamspot::BeamSpotContainer currentBS;


      // run based      
      if (beamSpotOutputBase_ == "runbased" ) {
	thisIOV = (cond::Time_t) iRun.id().run();
      }
      // lumi based
      else if (beamSpotOutputBase_ == "lumibased" ) {
	edm::LuminosityBlockID lu(iRun.id().run(),it->first);
	thisIOV = (cond::Time_t)(lu.value()); 

	currentBS.beamspot       = it -> second       ;
	currentBS.run            = iRun.id().run()    ;
	currentBS.beginLumiOfFit = it->first;
	currentBS.endLumiOfFit   = it->first;// to fix, for now endLumi = initLumi

	std::time_t lumi_t_begin = thisIOV; // to fix: meaningless value
	std::time_t lumi_t_end   = thisIOV; // to fix: meaningless value
	char ts[] = "yyyy.mn.dd hh:mm:ss zzz ";
	char* fbeginTime = ts;
	strftime(fbeginTime, sizeof(ts), "%Y.%m.%d %H:%M:%S GMT", gmtime(&lumi_t_begin));
	std::copy(fbeginTime, fbeginTime+32, currentBS.beginTimeOfFit);

	char* fendTime = ts;
	strftime(fendTime, sizeof(ts), "%Y.%m.%d %H:%M:%S GMT", gmtime(&lumi_t_end));
	std::copy(fendTime, fendTime+32, currentBS.endTimeOfFit);

	currentBS.reftime[0] = lumi_t_begin;
	currentBS.reftime[1] = lumi_t_end;

      }
      if (poolDbService->isNewTagRequest(outputrecordName_) ) {
          edm::LogInfo("AlcaBeamSpotHarvester")
              << "new tag requested" << std::endl;
          //poolDbService->createNewIOV<BeamSpotObjects>(aBeamSpot, poolDbService->beginOfTime(),poolDbService->endOfTime(),"BeamSpotObjectsRcd");
	  
	  //poolDbService->createNewIOV<BeamSpotObjects>(aBeamSpot, poolDbService->currentTime(), poolDbService->endOfTime(),"BeamSpotObjectsRcd");
	  poolDbService->writeOne<BeamSpotObjects>(aBeamSpot, thisIOV, outputrecordName_);
          if (dumpTxt_ && beamSpotOutputBase_ == "lumibased"){
              beamspot::dumpBeamSpotTxt(outTxt, false, currentBS);
          }    
      } 
      else {
        edm::LogInfo("AlcaBeamSpotHarvester")
            << "no new tag requested, appending IOV" << std::endl;
        //poolDbService->appendSinceTime<BeamSpotObjects>(aBeamSpot, poolDbService->currentTime(),"BeamSpotObjectsRcd");
	poolDbService->writeOne<BeamSpotObjects>(aBeamSpot, thisIOV, outputrecordName_);
        if (dumpTxt_ && beamSpotOutputBase_ == "lumibased"){
            beamspot::dumpBeamSpotTxt(outTxt, true, currentBS);
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
}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) {}

//--------------------------------------------------------------------------------------------------
void AlcaBeamSpotHarvester::endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup&) {
  theAlcaBeamSpotManager_.readLumi(iLumi);
}


DEFINE_FWK_MODULE(AlcaBeamSpotHarvester);
