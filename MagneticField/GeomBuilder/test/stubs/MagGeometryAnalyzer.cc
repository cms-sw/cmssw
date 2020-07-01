/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/VolumeBasedEngine/interface/MagGeometry.h"
#include "MagneticField/GeomBuilder/test/stubs/MagGeometryExerciser.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include <iostream>
#include <vector>

using namespace std;

class testMagGeometryAnalyzer : public edm::EDAnalyzer {
public:
  /// Constructor
  testMagGeometryAnalyzer(const edm::ParameterSet& pset){};

  /// Destructor
  ~testMagGeometryAnalyzer() override{};

  /// Perform the real analysis
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  void endJob() override {}

private:
  void testGrids(const vector<MagVolume6Faces const*>& bvol, const VolumeBasedMagneticField* field);
};

using namespace edm;

void testMagGeometryAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  ESHandle<MagneticField> magfield;
  eventSetup.get<IdealMagneticFieldRecord>().get(magfield);

  const VolumeBasedMagneticField* field = dynamic_cast<const VolumeBasedMagneticField*>(magfield.product());
  const MagGeometry* geom = field->field;

  // Test that findVolume succeeds for random points
  // This check is actually aleady covered by the standard regression.
  MagGeometryExerciser exe(geom);
  exe.testFindVolume(1000000);

  // Test that random points are inside one and only one volume.
  // Note: some overlaps are reported due to tolerance.
  // exe.testInside(100000,0.03);

  // Test that each grid point is inside its own volume
  // and check numerical problems in global volume search at volume boundaries.
  if (true) {
    cout << "***TEST GRIDS: barrel volumes: " << geom->barrelVolumes().size() << endl;
    testGrids(geom->barrelVolumes(), field);

    cout << "***TEST GRIDS: endcap volumes: " << geom->endcapVolumes().size() << endl;
    testGrids(geom->endcapVolumes(), field);
  }
}

#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "VolumeGridTester.h"

void testMagGeometryAnalyzer::testGrids(const vector<MagVolume6Faces const*>& bvol,
                                        const VolumeBasedMagneticField* field) {
  static map<string, int> nameCalls;

  for (vector<MagVolume6Faces const*>::const_iterator i = bvol.begin(); i != bvol.end(); i++) {
    if ((*i)->copyno != 1) {
      continue;
    }

    const MagProviderInterpol* prov = (**i).provider();
    if (prov == nullptr) {
      cout << (*i)->volumeNo << " No interpolator; skipping " << endl;
      continue;
    }
    VolumeGridTester tester(*i, prov, field);
    if (tester.testInside())
      cout << "testGrids: success: " << (**i).volumeNo << endl;
    else
      cout << "testGrids: ERROR: " << (**i).volumeNo << endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testMagGeometryAnalyzer);
