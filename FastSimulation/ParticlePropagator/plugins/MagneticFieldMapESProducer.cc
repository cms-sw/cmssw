#include "FastSimulation/ParticlePropagator/plugins/MagneticFieldMapESProducer.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

MagneticFieldMapESProducer::MagneticFieldMapESProducer(const edm::ParameterSet & p) 
{
    setWhatProduced(this);
    _label = p.getUntrackedParameter<std::string>("trackerGeometryLabel","");

    //    theTrackerMaterial = p.getParameter<edm::ParameterSet>("TrackerMaterial");

}

MagneticFieldMapESProducer::~MagneticFieldMapESProducer() {}

boost::shared_ptr<MagneticFieldMap> 
MagneticFieldMapESProducer::produce(const MagneticFieldMapRecord & iRecord){ 

  edm::ESHandle<TrackerInteractionGeometry> theInteractionGeometry;
  edm::ESHandle<MagneticField> theMagneticField;
  
  iRecord.getRecord<TrackerInteractionGeometryRecord>().get(_label, theInteractionGeometry );
  iRecord.getRecord<IdealMagneticFieldRecord>().get(theMagneticField );

  _map = boost::shared_ptr<MagneticFieldMap>
    (new MagneticFieldMap(&(*theMagneticField),&(*theInteractionGeometry)));

  return _map;

}


DEFINE_FWK_EVENTSETUP_MODULE(MagneticFieldMapESProducer);
