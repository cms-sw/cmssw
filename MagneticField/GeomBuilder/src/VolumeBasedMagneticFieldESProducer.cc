/** \file
 *
 *  $Date: 2006/10/27 01:35:34 $
 *  $Revision: 1.6 $
 */

#include "MagneticField/GeomBuilder/src/VolumeBasedMagneticFieldESProducer.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"

#include <string>
#include <iostream>

using namespace magneticfield;

VolumeBasedMagneticFieldESProducer::VolumeBasedMagneticFieldESProducer(const edm::ParameterSet& iConfig) : pset(iConfig)
{
  setWhatProduced(this);
}




// ------------ method called to produce the data  ------------
std::auto_ptr<MagneticField> VolumeBasedMagneticFieldESProducer::produce(const IdealMagneticFieldRecord & iRecord)
{
  edm::ESHandle<DDCompactView> cpv;
  iRecord.get("magfield",cpv );
  MagGeoBuilderFromDDD builder;
  builder.build(*cpv);
  
  std::auto_ptr<MagneticField> s(new VolumeBasedMagneticField(pset,
							      builder.barrelLayers(),
							      builder.endcapSectors(),
							      builder.barrelVolumes(),
							      builder.endcapVolumes()));
  return s;
}

DEFINE_FWK_EVENTSETUP_MODULE(VolumeBasedMagneticFieldESProducer);
