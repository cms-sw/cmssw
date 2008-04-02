/** \file
 *
 *  $Date: 2008/03/29 14:31:38 $
 *  $Revision: 1.10 $
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
  setWhatProduced(this, pset.getUntrackedParameter<std::string>("label",""));
}




// ------------ method called to produce the data  ------------
std::auto_ptr<MagneticField> VolumeBasedMagneticFieldESProducer::produce(const IdealMagneticFieldRecord & iRecord)
{
  edm::ESHandle<DDCompactView> cpv;
  iRecord.get("magfield",cpv );
  MagGeoBuilderFromDDD builder(pset.getParameter<std::string>("version"),
			       pset.getUntrackedParameter<bool>("debugBuilder", false));
  builder.build(*cpv);  

  // Get slave field
  edm::ESHandle<MagneticField> paramField;
  if (pset.getParameter<bool>("useParametrizedTrackerField")) {;
    iRecord.get("parametrizedField",paramField);
    //    std::cout << paramField->inTesla(GlobalPoint(0.,0.,0));
  }
  

  std::auto_ptr<MagneticField> s(new VolumeBasedMagneticField(pset,builder.barrelLayers(), builder.endcapSectors(), builder.barrelVolumes(), builder.endcapVolumes(), builder.maxR(), builder.maxZ(), paramField.product()));
  return s;
}

DEFINE_FWK_EVENTSETUP_MODULE(VolumeBasedMagneticFieldESProducer);
