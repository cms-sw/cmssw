/** \file
 *
 *  $Date: $
 *  $Revision: $
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

#include "Utilities/Timing/interface/TimingReport.h"
#include "MagneticField/Layers/interface/MagVerbosity.h"

//dirty hack
#define private public
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#undef public

#include <vector>
#include <map>
#include <string>

class MagGeometryAnalyzer : public edm::EDAnalyzer {
 public:
  /// Constructor
  MagGeometryAnalyzer(const edm::ParameterSet& pset) {};

  /// Destructor
  virtual ~MagGeometryAnalyzer() {};

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  virtual void endJob() {
    delete TimingReport::current();
  }
  
 private:

};

using namespace edm;

void MagGeometryAnalyzer::analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {

  ESHandle<MagneticField> magfield;
  eventSetup.get<IdealMagneticFieldRecord>().get(magfield);

//   edm::ParameterSet p;
//   p.addParameter<double>("findVolumeTolerance", 0.0);
//   p.addUntrackedParameter<bool>("cacheLastVolume", true);
//   p.addUntrackedParameter<bool>("timerOn", true);
//   MagGeoBuilderFromDDD builder;
//   MagGeometry* field = new MagGeometry(p,
// 				       builder.barrelLayers(),
// 				       builder.endcapSectors(),
// 				       builder.barrelVolumes(),
// 				       builder.endcapVolumes());

  const MagGeometry* field = (dynamic_cast<const VolumeBasedMagneticField*>(magfield.product()))->field;

  MagGeometryExerciser exe(field);
  exe.testFindVolume(100000);

  //  exe.testInside(100000);

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MagGeometryAnalyzer);
