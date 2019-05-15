/// CSCLayerGeometryInside.cc
/// Test CSCLayerGeometry::inside fiducial function
/// Tim Cox - 06.05.2009

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#include "TH2F.h"
#include "TFile.h"
#include "TRandom3.h"

#include <string>

class CSCLayerGeometryInside : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CSCLayerGeometryInside(const edm::ParameterSet&);
  ~CSCLayerGeometryInside() override;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  const std::string& myName() { return myName_; }

private:
  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;

  int ntries;  // no. of points to generate (set in config)

  // No,. of bins and range of x and y generation (set in config)
  int nbx;
  int nby;
  double xlo;
  double xhi;
  double ylo;
  double yhi;

  //      TH1F* h1as;
  //      TH1F* h1bs;

  TH2F* h1ai;
  TH2F* h1ao;
  TH2F* h1bi;
  TH2F* h1bo;
  TH2F* hall;

  TRandom3* tro;
};

CSCLayerGeometryInside::CSCLayerGeometryInside(const edm::ParameterSet& ps)
    : dashedLineWidth_(194),
      dashedLine_(std::string(dashedLineWidth_, '-')),
      myName_("CSCLayerGeometryInside"),
      ntries(ps.getUntrackedParameter<int>("ntries")),
      nbx(ps.getUntrackedParameter<int>("nbx")),
      nby(ps.getUntrackedParameter<int>("nby")),
      xlo(ps.getUntrackedParameter<double>("xlo")),
      xhi(ps.getUntrackedParameter<double>("xhi")),
      ylo(ps.getUntrackedParameter<double>("ylo")),
      yhi(ps.getUntrackedParameter<double>("yhi")) {
  usesResource("TFileService");

  std::cout << myName_ << " constructor:" << std::endl;
  std::cout << "x range is " << nbx << " bins from " << xlo << " to " << xhi << std::endl;
  std::cout << "y range is " << nby << " bins from " << ylo << " to " << yhi << std::endl;

  edm::Service<TFileService> fs;

  tro = new TRandom3();

  //  h1as = fs->make<TH1F>( "h1as", "ME1a strip", 800, 0., 80. );
  //  h1bs = fs->make<TH1F>( "h1bs", "ME1b strip", 800, 0., 80. );

  hall = fs->make<TH2F>("hall", "All      y vs x", nbx, xlo, xhi, nby, ylo, yhi);
  h1ai = fs->make<TH2F>("h1ai", "ME1a  in y vs x", nbx, xlo, xhi, nby, ylo, yhi);
  h1bi = fs->make<TH2F>("h1bi", "ME1b  in y vs x", nbx, xlo, xhi, nby, ylo, yhi);
  h1ao = fs->make<TH2F>("h1ao", "ME1a out y vs x", nbx, xlo, xhi, nby, ylo, yhi);
  h1bo = fs->make<TH2F>("h1bo", "ME1b out y vs x", nbx, xlo, xhi, nby, ylo, yhi);
}

CSCLayerGeometryInside::~CSCLayerGeometryInside() { delete tro; }

void CSCLayerGeometryInside::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::cout << myName() << ": Analyzer..." << std::endl;
  std::cout << "start " << dashedLine_ << std::endl;

  // Build the geometry
  edm::ESHandle<CSCGeometry> pgeom;
  iSetup.get<MuonGeometryRecord>().get(pgeom);

  std::cout << "CSCGeometry contains " << pgeom->chambers().size() << " chambers" << std::endl;
  std::cout << "CSCGeometry contains " << pgeom->layers().size() << " layers" << std::endl;

  std::cout << dashedLine_ << std::endl;

  // Construct an ME1a and a ME1b layer
  CSCDetId id1a(1, 1, 4, 1, 1);
  CSCDetId id1b(1, 1, 1, 1, 1);
  // Get a pointer to each layer
  const CSCLayer* p1a = pgeom->layer(id1a);
  const CSCLayer* p1b = pgeom->layer(id1b);
  // Get a pointer to each layer's layergeometry
  const CSCLayerGeometry* lg1a = p1a->geometry();
  const CSCLayerGeometry* lg1b = p1b->geometry();

  // Now generate a whole load of random LocalPoints and check whether they'e inside or outside the strip region

  //   float fstrips1a = lg1a->numberOfStrips();
  //   float fstrips1b = lg1b->numberOfStrips();
  //   float epsilon = 1.0e-06;

  for (int i = 1; i <= ntries; ++i) {
    float x = tro->Uniform(xlo, xhi);
    float y = tro->Uniform(ylo, yhi);
    LocalPoint lp(x, y);

    // Examine the float strip value returned by the StripTopology
    //     float strip1a = lg1a->topology()->strip( lp );
    //     float strip1b = lg1b->topology()->strip( lp );

    //     if ( strip1a > epsilon && strip1a < fstrips1a-epsilon ) h1as->Fill( strip1a );
    //     if ( strip1b > epsilon && strip1b < fstrips1b-epsilon ) h1bs->Fill( strip1b );

    // histogram each generated point
    hall->Fill(x, y);

    // histogram according to inside/outside 1a and 1b
    if (lg1a->inside(lp))
      h1ai->Fill(x, y);
    else
      h1ao->Fill(x, y);

    if (lg1b->inside(lp))
      h1bi->Fill(x, y);
    else
      h1bo->Fill(x, y);
  }

  std::cout << dashedLine_ << " end" << std::endl;
}

// This is a plug-in
DEFINE_FWK_MODULE(CSCLayerGeometryInside);
