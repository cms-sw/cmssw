#include "RecoLocalTracker/SiPixelRecHits/interface/PixelCPEFast.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

// new record 
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include <memory>

class  PixelCPEFastESProducer: public edm::ESProducer{
 public:
  PixelCPEFastESProducer(const edm::ParameterSet & p);
  std::shared_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord &);
 private:
  std::shared_ptr<PixelClusterParameterEstimator> cpe_;
  edm::ParameterSet pset_;
  edm::ESInputTag magname_;
  bool UseErrorsFromTemplates_;
};


#include <string>
#include <memory>

using namespace edm;




PixelCPEFastESProducer::PixelCPEFastESProducer(const edm::ParameterSet & p) 
{
  std::string myname = p.getParameter<std::string>("ComponentName");
  magname_ = p.existsAs<edm::ESInputTag>("MagneticFieldRecord")?
    p.getParameter<edm::ESInputTag>("MagneticFieldRecord"):edm::ESInputTag("");
  UseErrorsFromTemplates_    = p.getParameter<bool>("UseErrorsFromTemplates");


  pset_ = p;
  setWhatProduced(this,myname);


}


std::shared_ptr<PixelClusterParameterEstimator>
PixelCPEFastESProducer::produce(const TkPixelCPERecord & iRecord){ 

  ESHandle<MagneticField> magfield;
  iRecord.getRecord<IdealMagneticFieldRecord>().get( magname_, magfield );

  edm::ESHandle<TrackerGeometry> pDD;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get( pDD );

  edm::ESHandle<TrackerTopology> hTT;
  iRecord.getRecord<TrackerDigiGeometryRecord>().getRecord<TrackerTopologyRcd>().get(hTT);

  // Lorant angle for offsets
  ESHandle<SiPixelLorentzAngle> lorentzAngle;
  iRecord.getRecord<SiPixelLorentzAngleRcd>().get(lorentzAngle );

  // add the new la width object
  ESHandle<SiPixelLorentzAngle> lorentzAngleWidth;
  const SiPixelLorentzAngle * lorentzAngleWidthProduct = nullptr;
  iRecord.getRecord<SiPixelLorentzAngleRcd>().get("forWidth",lorentzAngleWidth );
  lorentzAngleWidthProduct = lorentzAngleWidth.product();

  const SiPixelGenErrorDBObject * genErrorDBObjectProduct = nullptr;

  // Errors take only from new GenError
  ESHandle<SiPixelGenErrorDBObject> genErrorDBObject;
  if(UseErrorsFromTemplates_) {  // do only when generrors are needed
    iRecord.getRecord<SiPixelGenErrorDBObjectRcd>().get(genErrorDBObject); 
    genErrorDBObjectProduct = genErrorDBObject.product();
    //} else {
    //std::cout<<" pass an empty GenError pointer"<<std::endl;
  }
  cpe_  = std::make_shared<PixelCPEFast>(
                         pset_,magfield.product(),*pDD.product(),
			 *hTT.product(),lorentzAngle.product(),
			 genErrorDBObjectProduct,lorentzAngleWidthProduct);

  return cpe_;
}


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"

DEFINE_FWK_EVENTSETUP_MODULE(PixelCPEFastESProducer);

