#ifndef BTRANSITIONANALYZER_H
#define BTRANSITIONANALYZER_H

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

namespace cond {
  template<class T, class R>
  class BTransitionAnalyzer: public edm::one::EDAnalyzer<edm::one::WatchRuns> {
  public:
    BTransitionAnalyzer( const edm::ParameterSet& pset ):
      m_currentThreshold( pset.getUntrackedParameter<double>( "currentThreshold", 18000. ) ) {
    }
#ifdef __INTEL_COMPILER
    virtual ~BTransitionAnalyzer() = default;
#endif
    // implicit copy constructor
    // implicit assignment operator
    // implicit destructor
    virtual void beginJob() final {};
    virtual void beginRun(edm::Run const&, edm::EventSetup const&) final {};
    virtual void analyze(edm::Event const&, edm::EventSetup const&) final {};
    virtual void endRun(edm::Run const& run , edm::EventSetup const& eventSetup) final {
      edm::ESHandle<RunInfo> runInfoHandle;
      edm::ESHandle<T> payloadHandle, payloadRefHandle;
      eventSetup.get<RunInfoRcd>().get(runInfoHandle);
      double avg_current = (double)runInfoHandle->m_avg_current;
      double current_default = -1;
      std::string bOnLabel = std::string( "38T" );
      std::string bOffLabel = std::string( "0T" );
      std::string bFieldLabel = bOnLabel;
      LogDebug( "BTransitionAnalyzer" ) << "Comparing value of magnet current: " << avg_current << " A for run: " << run.run()
                                        << " with the corresponding threshold: "<< m_currentThreshold << " A." << std::endl;
      if( avg_current != current_default && avg_current <= m_currentThreshold ) bFieldLabel = bOffLabel;
      edm::LogInfo( "BTransitionAnalyzer" ) << "The magnet was " 
                                            << ( bFieldLabel == bOnLabel ? "ON" : "OFF" ) << " during run " << run.run()
                                            << ".\nLoading the product for the corrisponding label " << bFieldLabel << std::endl;
      eventSetup.get<R>().get( bFieldLabel, payloadHandle );
      eventSetup.get<R>().get( payloadRefHandle );
      edm::Service<cond::service::PoolDBOutputService> mydbservice;
      if( mydbservice.isAvailable() ) {
        if( !equalPayloads( payloadHandle, payloadRefHandle ) ) {
          edm::LogInfo( "BTransitionAnalyzer" ) << "Exporting payload corresponding to the calibrations for magnetic field "
                                                << ( bFieldLabel == bOnLabel ? "ON" : "OFF" )
                                                << " starting from run number: " << run.run() << std::endl;
          mydbservice->writeOne( payloadHandle.product(), run.run(), demangledName( typeid(R) ) );
        } else {
          edm::LogInfo( "BTransitionAnalyzer" ) << "The payload corresponding to the calibrations for magnetic field "
                                                << ( bFieldLabel == bOnLabel ? "ON" : "OFF" )
                                                << " is still valid for run " << run.run()
                                                << ".\nNo transfer needed." << std::endl;
        }
      } else {
        edm::LogError( "BTransitionAnalyzer" ) << "PoolDBOutputService unavailable";
      }
    }
    virtual void endJob() final {};
    virtual bool equalPayloads( edm::ESHandle<T> const & payloadHandle, edm::ESHandle<T> const & payloadRefHandle ) = 0;
  private:
    double m_currentThreshold;
  };
} //namespace cond
#endif //BTRANSITIONANALYZER_H
