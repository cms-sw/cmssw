#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/RegressionTest/src/DDCheck.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/FIPConfiguration.h"
#include "DetectorDescription/RegressionTest/interface/DDCompareTools.h"

class CompareDDCompactViews : public edm::one::EDAnalyzer<edm::one::WatchRuns>
{
public:
  explicit CompareDDCompactViews( const edm::ParameterSet& );
  ~CompareDDCompactViews() override {}
  
  void beginJob() override {}
  void beginRun( edm::Run const& , edm::EventSetup const& ) override;
  void analyze( edm::Event const& , edm::EventSetup const& ) override {}
  void endRun( edm::Run const& , edm::EventSetup const& ) override {}
  void endJob() override {}

private:
  std::string m_fname1;
  std::string m_fname2;
};

CompareDDCompactViews::CompareDDCompactViews( const edm::ParameterSet& iConfig )
{
  m_fname1 = iConfig.getUntrackedParameter<std::string>( "XMLFileName1" );
  m_fname2 = iConfig.getUntrackedParameter<std::string>( "XMLFileName2" );
}

void
CompareDDCompactViews::beginRun( const edm::Run&, edm::EventSetup const& es ) 
{
  DDCompactView cpv1;
  DDLParser parser1( cpv1 );
  parser1.parseOneFile( m_fname1 );
  DDCheckMaterials( std::cout );
  cpv1.lockdown();

  DDCompactView cpv2;
  DDLParser parser2( cpv2 );
  parser2.parseOneFile( m_fname2 );
  DDCheckMaterials( std::cout );
  cpv2.lockdown();

  DDCompOptions ddco;
  DDCompareCPV ddccpv( ddco );
  bool graphmatch = ddccpv( cpv1, cpv2 );
   
  if( graphmatch ) {
    std::cout << "DDCompactView graphs match" << std::endl;
  } else {
    std::cout << "DDCompactView graphs do NOT match" << std::endl;
  }
}

DEFINE_FWK_MODULE( CompareDDCompactViews );
