#ifndef Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H
#define Geometry_TrackerNumberingBuilder_TrackerGeometricDetESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDetExtra.h"

class  TrackerGeometricDetESModule: public edm::ESProducer,
				    public edm::EventSetupRecordIntervalFinder {
 public:
  TrackerGeometricDetESModule(const edm::ParameterSet & p);
  virtual ~TrackerGeometricDetESModule(); 
  std::auto_ptr<GeometricDet>       produce(const IdealGeometryRecord &);
  boost::shared_ptr<std::vector<GeometricDetExtra> > produceGDE(const IdealGeometryRecord &);

 protected:
  //overriding from ContextRecordIntervalFinder
  virtual void setIntervalFor( const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue& ,
                               edm::ValidityInterval& ) ;

 private:
  void putOne(std::vector<GeometricDetExtra> & gde, const GeometricDet* gd, const DDExpandedView& ev, int lev );

  bool fromDDD_;
  // try without this first.  std::vector<GeometricDetExtra> gdv_;

};


#endif




