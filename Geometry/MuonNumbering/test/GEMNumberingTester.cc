#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/MuonNumbering/interface/GEMNumberingScheme.h"

class GEMNumberingTester : public edm::EDAnalyzer
{
public:
  explicit GEMNumberingTester( const edm::ParameterSet& );
  ~GEMNumberingTester( void );

  virtual void analyze( const edm::Event&, const edm::EventSetup& );
};

GEMNumberingTester::GEMNumberingTester( const edm::ParameterSet& iConfig )
{
}

GEMNumberingTester::~GEMNumberingTester( void )
{
}

void
GEMNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  using namespace edm;

  ESTransientHandle<DDCompactView> cv;
  iSetup.get<IdealGeometryRecord>().get( cv );

  MuonNumberingScheme* numbering = new GEMNumberingScheme( *cv );
  delete numbering;
}

DEFINE_FWK_MODULE( GEMNumberingTester );
