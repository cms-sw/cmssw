#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DetectorDescription/DDCMS/interface/DDRegistry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "base/Transformation3D.h"
#include "navigation/NavigationState.h"
#include "navigation/SimpleNavigator.h"
#include "volumes/LogicalVolume.h"
#include "volumes/Box.h"

#include <memory>
#include <string>

using namespace vecgeom;

class DDVecCMSDetector : public edm::one::EDAnalyzer<> {
public:
  explicit DDVecCMSDetector(const edm::ParameterSet& p);

  void beginJob() override {}
  void analyze( edm::Event const& iEvent, edm::EventSetup const& ) override;
  void endJob() override;
};

DDVecCMSDetector::DDVecCMSDetector( const edm::ParameterSet& )
{}

void
DDVecCMSDetector::analyze( const edm::Event&, const edm::EventSetup& )
{
  UnplacedBox world_params    = UnplacedBox(4., 4., 4.);
  UnplacedBox largebox_params = UnplacedBox(1.5, 1.5, 1.5);
  UnplacedBox smallbox_params = UnplacedBox(0.5, 0.5, 0.5);

  LogicalVolume worldl(&world_params);
  LogicalVolume largebox("Large box", &largebox_params);
  LogicalVolume smallbox("Small box", &smallbox_params);

  Transformation3D origin     = Transformation3D();
  Transformation3D placement1 = Transformation3D(2, 2, 2);
  Transformation3D placement2 = Transformation3D(-2, 2, 2);
  Transformation3D placement3 = Transformation3D(2, -2, 2);
  Transformation3D placement4 = Transformation3D(2, 2, -2);
  Transformation3D placement5 = Transformation3D(-2, -2, 2);
  Transformation3D placement6 = Transformation3D(-2, 2, -2);
  Transformation3D placement7 = Transformation3D(2, -2, -2);
  Transformation3D placement8 = Transformation3D(-2, -2, -2);

  largebox.PlaceDaughter(&smallbox, &origin);
  worldl.PlaceDaughter(&largebox, &placement1);
  worldl.PlaceDaughter(&largebox, &placement2);
  worldl.PlaceDaughter(&largebox, &placement3);
  worldl.PlaceDaughter(&largebox, &placement4);
  worldl.PlaceDaughter("Hello the world!", &largebox, &placement5);
  worldl.PlaceDaughter(&largebox, &placement6);
  worldl.PlaceDaughter(&largebox, &placement7);
  worldl.PlaceDaughter(&largebox, &placement8);

  VPlacedVolume *world_placed = worldl.Place();
  GeoManager::Instance().SetWorld(world_placed);
  GeoManager::Instance().CloseGeometry();

  std::cerr << "Printing world content:\n";
  world_placed->PrintContent();

  SimpleNavigator nav;
  Vector3D<Precision> point(2, 2, 2);
  NavigationState *path = NavigationState::MakeInstance(4);
  nav.LocatePoint(world_placed, point, *path, true);
  path->Print();

  GeoManager::Instance().FindLogicalVolume("Large box");
  GeoManager::Instance().FindPlacedVolume("Large box");

  NavigationState::ReleaseInstance(path);
}

void
DDVecCMSDetector::endJob()
{}

DEFINE_FWK_MODULE( DDVecCMSDetector );
