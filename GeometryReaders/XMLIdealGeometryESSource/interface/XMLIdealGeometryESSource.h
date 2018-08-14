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

class XMLIdealGeometryESSource : public edm::ESProducer, 
                                 public edm::EventSetupRecordIntervalFinder
{
public:
    XMLIdealGeometryESSource(const edm::ParameterSet & p);
    ~XMLIdealGeometryESSource() override; 
    std::unique_ptr<DDCompactView> produceGeom(const IdealGeometryRecord &);
    std::unique_ptr<DDCompactView> produceMagField(const IdealMagneticFieldRecord &);
    std::unique_ptr<DDCompactView> produce();
protected:
    void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
			const edm::IOVSyncValue &,edm::ValidityInterval &) override;
    XMLIdealGeometryESSource(const XMLIdealGeometryESSource &) = delete;
    const XMLIdealGeometryESSource & operator=(const XMLIdealGeometryESSource &) = delete;

 private:
    std::string rootNodeName_;
    bool userNS_;
    GeometryConfiguration geoConfig_;
};


#endif
