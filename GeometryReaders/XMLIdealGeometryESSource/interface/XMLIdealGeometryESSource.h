#ifndef GeometryReaders_XMLIdealGeometryESSource_XMLIdealGeometryESSource_H
#define GeometryReaders_XMLIdealGeometryESSource_XMLIdealGeometryESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "GeometryReaders/XMLIdealGeometryESSource/interface/GeometryConfiguration.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include <memory>
#include <string>

namespace DDI {
  class Material;
  class Solid;
  class LogicalPart;
  class Specific;
}

class XMLIdealGeometryESSource : public edm::ESProducer, 
                                 public edm::EventSetupRecordIntervalFinder
{
public:
    XMLIdealGeometryESSource(const edm::ParameterSet & p);
    virtual ~XMLIdealGeometryESSource(); 
    std::auto_ptr<DDCompactView> produceGeom(const IdealGeometryRecord &);
    std::auto_ptr<DDCompactView> produceMagField(const IdealMagneticFieldRecord &);
    std::auto_ptr<DDCompactView> produce();
protected:
    virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
				const edm::IOVSyncValue &,edm::ValidityInterval &);
private:
    XMLIdealGeometryESSource(const XMLIdealGeometryESSource &);
    const XMLIdealGeometryESSource & operator=(const XMLIdealGeometryESSource &);
    std::string rootNodeName_;
    bool userNS_,cpvavailable_;
    GeometryConfiguration geoConfig_;

    // 2009-07-09 memory patch
    // for copying and protecting DD Store's after parsing is complete.
    DDI::Store<DDName, DDI::Material*>::registry_type matStore_;
    DDI::Store<DDName, DDI::Solid*>::registry_type solidStore_;
    DDI::Store<DDName, DDI::LogicalPart*>::registry_type lpStore_;
    DDI::Store<DDName, DDI::Specific*>::registry_type specStore_;
    DDI::Store<DDName, DDRotationMatrix*>::registry_type rotStore_;    
};


#endif
