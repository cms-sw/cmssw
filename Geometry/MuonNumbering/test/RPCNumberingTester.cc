#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/RPCNumberingScheme.h"

class RPCNumberingTester : public edm::EDAnalyzer
{
public:
  explicit RPCNumberingTester( const edm::ParameterSet& );
  ~RPCNumberingTester( void );

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
};

RPCNumberingTester::RPCNumberingTester( const edm::ParameterSet& iConfig )
{
}

RPCNumberingTester::~RPCNumberingTester( void )
{
}

void
RPCNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;

  ESTransientHandle<DDCompactView> cv;
  iSetup.get<IdealGeometryRecord>().get( cv );

  MuonNumberingScheme* numbering = new RPCNumberingScheme( *cv );
  delete numbering;
}

DEFINE_FWK_MODULE( RPCNumberingTester );
