// -*- C++ -*-
//
// Class:      WriteCTPPSBeamParameters
//
// Description: Test analyzer for CTPPS beam parameters condition data
//
//              Simple analyzer that writes one CTTPSBeamParameters record into a sql
//              database file, as a test of offline conditions implementation.
//              Another analyzer is then used to retrieve these conditions.
//
// Original Author:  Wagner De Paula Carvalho
//         Created:  Wed, 21 Nov 2018 17:35:07 GMT
//
//==================================================================================

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSBeamParameters.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"

#include <stdint.h>

class WriteCTPPSBeamParameters : public edm::one::EDAnalyzer<>
{
  public:
    WriteCTPPSBeamParameters(const edm::ParameterSet&) {}
    ~WriteCTPPSBeamParameters() = default;

  private:
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
};

//---------------------------------------------------------------------------------------

void WriteCTPPSBeamParameters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::ESHandle<CTPPSBeamParameters> bp ;
  iSetup.get<CTPPSBeamParametersRcd>().get(bp) ;

  // Pointer for the conditions data object
  const CTPPSBeamParameters *p = bp.product() ;

  // Using "lumiid" as IOV
  const edm::LuminosityBlock &iLBlock = iEvent.getLuminosityBlock() ;
  edm::LuminosityBlockID lu(iLBlock.run(), iLBlock.id().luminosityBlock()) ;
  cond::Time_t ilumi = (cond::Time_t)(lu.value()) ;
  // cond::Time_t itime = (cond::Time_t)(iEvent.time().value()) ;  //  use this for timestamp

  edm::LogInfo("WriteCTPPSBeamParameters::analyze") << "cond::Time_t ilumi = " << ilumi
    << " = " << boost::posix_time::to_iso_extended_string( cond::time::to_boost( ilumi ) ) << "\n" ;

  // Write to database or sqlite file
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if( poolDbService.isAvailable() )
    poolDbService->writeOne( p, ilumi, "CTPPSBeamParametersRcd"  );
    // poolDbService->writeOne( p, poolDbService->currentTime(), "CTPPSBeamParametersRcd"  );
  else
    throw std::runtime_error("PoolDBService required.");
}

//define this as a plug-in
DEFINE_FWK_MODULE(WriteCTPPSBeamParameters);

