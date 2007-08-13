#ifndef Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H
#define Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/DataRecord/interface/PGeometricDetRcd.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

class  TrackerGeometricDetESModule: public edm::ESProducer,
				    public edm::EventSetupRecordIntervalFinder {
 public:
  TrackerGeometricDetESModule(const edm::ParameterSet & p);
  virtual ~TrackerGeometricDetESModule(); 
  std::auto_ptr<GeometricDet>  produceFromDDDXML(const IdealGeometryRecord &);
  std::auto_ptr<GeometricDet>  produceFromPGeometricDet(const PGeometricDetRcd &);

 protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue& ,
                               edm::ValidityInterval& ) ;

 private:
  bool fromDDD_;

};


#endif




