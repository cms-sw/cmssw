/****************************************************************************
 *
 * Authors:
 *  Jan Kaspar
 * Adapted by:
 *  Helena Malbouisson
 *  Clemencia Mora Herrera  
 ****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "CondCore/CondDB/interface/Time.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"


#include "CondFormats/AlignmentRecord/interface/CTPPSRPAlignmentCorrectionsDataRcd.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Class to print out information on current geometry.
 **/
class CTPPSRPAlignmentInfoAnalyzer : public edm::one::EDAnalyzer<>
{
public:
  CTPPSRPAlignmentInfoAnalyzer( const edm::ParameterSet& ps);
  ~CTPPSRPAlignmentInfoAnalyzer() override{}

private: 
  void analyze( const edm::Event& e, const edm::EventSetup& es ) override;

  void printInfo(const CTPPSRPAlignmentCorrectionsData &alignments, const edm::Event& event) const;
  edm::ESWatcher<CTPPSRPAlignmentCorrectionsDataRcd> watcherAlignments_;

  cond::Time_t iov_;
  std::string record_;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSRPAlignmentInfoAnalyzer::CTPPSRPAlignmentInfoAnalyzer( const edm::ParameterSet& iConfig )
 {
   record_=iConfig.getParameter<string>("record");
   iov_=iConfig.getParameter<unsigned long long>("iov");
}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentInfoAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  
  edm::ESHandle<CTPPSRPAlignmentCorrectionsData> alignments;
  if ( watcherAlignments_.check( iSetup ) )
    {
      
      iSetup.get<CTPPSRPAlignmentCorrectionsDataRcd>().get(alignments);
      
      const CTPPSRPAlignmentCorrectionsData* pCTPPSRPAlignmentCorrectionsData = alignments.product();
      edm::Service<cond::service::PoolDBOutputService> poolDbService;
      if( poolDbService.isAvailable() ){
	poolDbService->writeOne( pCTPPSRPAlignmentCorrectionsData, iov_,  record_  );
      }
      
    }
  return;

}

//----------------------------------------------------------------------------------------------------

void CTPPSRPAlignmentInfoAnalyzer::printInfo(const CTPPSRPAlignmentCorrectionsData &alignments, const edm::Event& event) const
{
  time_t unixTime = event.time().unixTime();
  char timeStr[50];
  strftime( timeStr, 50, "%F %T", localtime( &unixTime ) );

  edm::LogInfo("CTPPSRPAlignmentInfoAnalyzer")
    << "New  alignments found in run="
    << event.id().run() << ", event=" << event.id().event() << ", UNIX timestamp=" << unixTime
    << " (" << timeStr << "):\n"
    << alignments;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSRPAlignmentInfoAnalyzer );
