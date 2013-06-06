#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/DTNumberingScheme.h"

class DTNumberingTester : public edm::EDAnalyzer
{
public:
  explicit DTNumberingTester( const edm::ParameterSet& );
  ~DTNumberingTester( void );

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
};

DTNumberingTester::DTNumberingTester( const edm::ParameterSet& iConfig )
{
}

DTNumberingTester::~DTNumberingTester( void )
{
}

void
DTNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;

  ESTransientHandle<DDCompactView> cv;
  iSetup.get<IdealGeometryRecord>().get( cv );

  MuonNumberingScheme* numbering = new DTNumberingScheme( *cv );
  delete numbering;
}

DEFINE_FWK_MODULE( DTNumberingTester );
