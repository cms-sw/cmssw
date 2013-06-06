#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/CSCNumberingScheme.h"

class CSCNumberingTester : public edm::EDAnalyzer
{
public:
  explicit CSCNumberingTester( const edm::ParameterSet& );
  ~CSCNumberingTester( void );

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
};

CSCNumberingTester::CSCNumberingTester( const edm::ParameterSet& iConfig )
{
}

CSCNumberingTester::~CSCNumberingTester( void )
{
}

void
CSCNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;

  ESTransientHandle<DDCompactView> cv;
  iSetup.get<IdealGeometryRecord>().get( cv );

  MuonNumberingScheme* numbering = new CSCNumberingScheme( *cv );
  delete numbering;
}

DEFINE_FWK_MODULE( CSCNumberingTester );
