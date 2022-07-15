/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/VolumeBasedEngine/interface/MagGeometry.h"
#include "MagneticField/GeomBuilder/test/stubs/MagGeometryExerciser.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include <iostream>
#include <vector>

class testMagGeometryAnalyzer : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  testMagGeometryAnalyzer(const edm::ParameterSet& pset);

  /// Destructor
  ~testMagGeometryAnalyzer() override = default;

  /// Perform the real analysis
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  void endJob() override {}

private:
  void testGrids(const std::vector<MagVolume6Faces const*>& bvol, const VolumeBasedMagneticField* field);

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
};

testMagGeometryAnalyzer::testMagGeometryAnalyzer(const edm::ParameterSet&)
    : magfieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()) {}

void testMagGeometryAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  const edm::ESHandle<MagneticField>& magfield = eventSetup.getHandle(magfieldToken_);

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
    edm::LogVerbatim("MagGeometry") << "***TEST GRIDS: barrel volumes: " << geom->barrelVolumes().size();
    testGrids(geom->barrelVolumes(), field);

    edm::LogVerbatim("MagGeometry") << "***TEST GRIDS: endcap volumes: " << geom->endcapVolumes().size();
    testGrids(geom->endcapVolumes(), field);
  }
}

#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "VolumeGridTester.h"

void testMagGeometryAnalyzer::testGrids(const std::vector<MagVolume6Faces const*>& bvol,
                                        const VolumeBasedMagneticField* field) {
  static std::map<std::string, int> nameCalls;

  for (std::vector<MagVolume6Faces const*>::const_iterator i = bvol.begin(); i != bvol.end(); i++) {
    if ((*i)->copyno != 1) {
      continue;
    }

    const MagProviderInterpol* prov = (**i).provider();
    if (prov == 0) {
      edm::LogVerbatim("MagGeometry") << (*i)->volumeNo << " No interpolator; skipping ";
      continue;
    }
    VolumeGridTester tester(*i, prov, field);
    if (tester.testInside())
      edm::LogVerbatim("MagGeometry") << "testGrids: success: " << (**i).volumeNo;
    else
      edm::LogVerbatim("MagGeometry") << "testGrids: ERROR: " << (**i).volumeNo;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testMagGeometryAnalyzer);
