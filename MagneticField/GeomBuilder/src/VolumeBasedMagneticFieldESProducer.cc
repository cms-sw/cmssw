// user include files
#include "MagneticField/GeomBuilder/interface/VolumeBasedMagneticFieldESProducer.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"

#include <string>
#include <iostream>

using namespace magneticfield;

VolumeBasedMagneticFieldESProducer::VolumeBasedMagneticFieldESProducer(const edm::ParameterSet& /*iConfig*/)
{
  setWhatProduced(this);
}




// ------------ method called to produce the data  ------------
std::auto_ptr<MagneticField> VolumeBasedMagneticFieldESProducer::produce(const IdealMagneticFieldRecord & iRecord)
{
  edm::ESHandle<DDCompactView> cpv;
  iRecord.getRecord<IdealGeometryRecord>().get("magfield",cpv );
  MagGeoBuilderFromDDD builder;
  builder.build(*cpv);
  
  std::auto_ptr<MagneticField> s(new VolumeBasedMagneticField(builder.barrelLayers(),
									 builder.endcapSectors(),
									 builder.barrelVolumes(),
									 builder.endcapVolumes()));
  return s;
}

DEFINE_FWK_EVENTSETUP_MODULE(VolumeBasedMagneticFieldESProducer)
