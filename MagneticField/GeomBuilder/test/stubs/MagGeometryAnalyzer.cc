/** \file
 *
 *  $Date: 2013/04/15 16:22:20 $
 *  $Revision: 1.8 $
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

#include "MagneticField/Layers/interface/MagVerbosity.h"

//dirty hack
#define private public
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#undef public

#include <iostream>
#include <vector>

using namespace std;

class MagGeometryAnalyzer : public edm::EDAnalyzer {
 public:
  /// Constructor
  MagGeometryAnalyzer(const edm::ParameterSet& pset) {};

  /// Destructor
  virtual ~MagGeometryAnalyzer() {};

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void endJob() {
  }
  
 private:
  void testGrids( const vector<MagVolume6Faces*>& bvol);
};

using namespace edm;

void MagGeometryAnalyzer::analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {

  ESHandle<MagneticField> magfield;
  eventSetup.get<IdealMagneticFieldRecord>().get(magfield);

  const MagGeometry* field = (dynamic_cast<const VolumeBasedMagneticField*>(magfield.product()))->field;
  
  
  // Test that findVolume succeeds for random points
  MagGeometryExerciser exe(field);

  //FIXME: the region to be tested is specified inside.
  exe.testFindVolume(10000000);

  // Test that random points are inside one and only one volume
  // exe.testInside(100000,0.03); 

  
  // Test that each grid point is inside its own volume
  if (false) {
    cout << "***TEST GRIDS: barrel volumes: " << field->barrelVolumes().size() << endl;
    testGrids( field->barrelVolumes());
    
    cout << "***TEST GRIDS: endcap volumes: " << field->endcapVolumes().size() << endl;
    testGrids( field->endcapVolumes());
  }
}


#include "MagneticField/VolumeGeometry/interface/MagVolume6Faces.h"
#include "VolumeGridTester.h"


void MagGeometryAnalyzer::testGrids(const vector<MagVolume6Faces*>& bvol) {
  static map<string,int> nameCalls;

  for (vector<MagVolume6Faces*>::const_iterator i=bvol.begin();
       i!=bvol.end(); i++) {
    if ((*i)->copyno != 1) {
      continue;
    }

    const MagProviderInterpol* prov = (**i).provider();
    if (prov == 0) {
      cout << (*i)->volumeNo << " No interpolator; skipping " <<  endl;
      continue;
    }
    VolumeGridTester tester(*i, prov);
    if (tester.testInside()) cout << "testGrids: success: " << (**i).volumeNo << endl;
    else cout << "testGrids: ERROR: " << (**i).volumeNo << endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MagGeometryAnalyzer);
