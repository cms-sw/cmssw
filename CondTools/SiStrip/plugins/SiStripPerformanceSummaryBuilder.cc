#include "CondTools/SiStrip/plugins/SiStripPerformanceSummaryBuilder.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include <iostream>
#include <fstream>

SiStripPerformanceSummaryBuilder::SiStripPerformanceSummaryBuilder(const edm::ParameterSet& iConfig):
  fp_(iConfig.getUntrackedParameter<edm::FileInPath>("file",edm::FileInPath("CalibTracker/SiStripCommon/data/SiStripDetInfo.dat"))),
  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

void SiStripPerformanceSummaryBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  SiStripPerformanceSummary* psummary = new SiStripPerformanceSummary();
  // fill object

  SiStripDetInfoFileReader reader(fp_.fullPath());

 
  for(std::vector<uint32_t>::const_iterator idet = reader.getAllDetIds().begin(); idet != reader.getAllDetIds().end(); ++idet){
    // generate random values for each detId
    float clusterSizeMean = (float) RandGauss::shoot(4.,2.);
    float clusterSizeRMS  = (float) RandGauss::shoot(2.,1.);
    float clusterChargeMean = (float) RandGauss::shoot(70.,10.);
    float clusterChargeRMS  = (float) RandGauss::shoot(10.,1.);
    float occupancyMean = (float) RandGauss::shoot(50.,20.);
    float occupancyRMS  = (float) RandGauss::shoot(20.,4.);
    float noisyStrips = (float) RandGauss::shoot(7.,1.);
    // set values
    psummary->setClusterSize(*idet, clusterSizeMean, clusterSizeRMS);
    psummary->setClusterCharge(*idet, clusterChargeMean, clusterChargeRMS);
    psummary->setOccupancy(*idet, occupancyMean, occupancyRMS);
    psummary->setPercentNoisyStrips(*idet, noisyStrips);
  }
  clock_t presentTime = clock();
  psummary->setTimeValue((unsigned long long)presentTime);
  psummary->print();
  // Write to DB
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() ){
    if ( poolDbService->isNewTagRequest( "SiStripPerformanceSummaryRcd" ) ){
      edm::LogInfo("Tag")<<" is new tag request.";
      poolDbService->createNewIOV<SiStripPerformanceSummary>( psummary, poolDbService->beginOfTime(),poolDbService->endOfTime(),"SiStripPerformanceSummaryRcd"  );
    }else{
      edm::LogInfo("Tag")<<" tag exists already.";
      poolDbService->appendSinceTime<SiStripPerformanceSummary>( psummary, poolDbService->currentTime(),"SiStripPerformanceSummaryRcd" );
    }
  }else{
    edm::LogError("PoolDBOutputService")<<" Service is unavailable"<<std::endl;
  }
}

