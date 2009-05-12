#include "RecoLocalTracker/SiStripRecHitConverter/plugins/StripCPEESProducer.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPE.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEfromTrackAngle2.h"
#include "RecoLocalTracker/SiStripRecHitConverter/interface/StripCPEgeometric.h"
#include "FWCore/Framework/interface/ESHandle.h"

StripCPEESProducer::StripCPEESProducer(const edm::ParameterSet & p) 
{
  name_ = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,name_);
}

boost::shared_ptr<StripClusterParameterEstimator> StripCPEESProducer::
produce(const TkStripCPERecord & iRecord) 
{ 
  edm::ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(magfield );
  
  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );
  
  edm::ESHandle<SiStripLorentzAngle> lorentzAngle;
  iRecord.getRecord<SiStripLorentzAngleRcd>().get(lorentzAngle);

  if(name_=="SimpleStripCPE") 
    cpe_ = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPE(               pset_, magfield.product(), pDD.product(), lorentzAngle.product() ));  
  else if(name_=="StripCPEfromTrackAngle")
    cpe_ = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTrackAngle( pset_, magfield.product(), pDD.product(), lorentzAngle.product() ));
  else if(name_=="StripCPEfromTrackAngle2")
    cpe_ = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEfromTrackAngle2(pset_, magfield.product(), pDD.product(), lorentzAngle.product() ));
  else if(name_=="StripCPEgeometric")
    cpe_ = boost::shared_ptr<StripClusterParameterEstimator>(new StripCPEgeometric(pset_, magfield.product(), pDD.product(), lorentzAngle.product() ));
  else throw cms::Exception("Unknown StripCPE type") << name_;

  return cpe_;
}
