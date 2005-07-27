#ifndef GeometryReaders_XMLIdealGeometryESSource_XMLIdealGeometryESSource_H
#define GeometryReaders_XMLIdealGeometryESSource_XMLIdealGeometryESSource_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <string>

class XMLIdealGeometryESSource : public edm::eventsetup::ESProducer, 
                                  public edm::eventsetup::EventSetupRecordIntervalFinder
{
public:
    XMLIdealGeometryESSource(const edm::ParameterSet & p);
    virtual ~XMLIdealGeometryESSource(); 
    const DDCompactView * produce(const IdealGeometryRecord &);
protected:
    virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
				const edm::Timestamp &,edm::ValidityInterval &);
private:
    XMLIdealGeometryESSource(const XMLIdealGeometryESSource &);
    const XMLIdealGeometryESSource & operator=(const XMLIdealGeometryESSource &);
};


#endif
