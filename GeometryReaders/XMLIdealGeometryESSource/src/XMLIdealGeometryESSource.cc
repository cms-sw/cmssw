#include "GeometryReaders/XMLIdealGeometryESSource/interface/XMLIdealGeometryESSource.h"

#include "DetectorDescription/Core/interface/DDdebug.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"

#include <memory>

using namespace edm::eventsetup;

XMLIdealGeometryESSource::XMLIdealGeometryESSource(const edm::ParameterSet & p) 
{
    DDLParser * parser = DDLParser::instance();
    AlgoInit();
    DDLConfiguration cf;
    int result1 = cf.readConfig(p.getParameter<std::string>("GeometryConfiguration"));
    if (result1 !=0) throw DDException("DDLConfiguration: readConfig failed !");
    int result2 = parser->parse(cf);
    if (result2 != 0) throw DDException("DDD-Parser: parsing failed!");
    setWhatProduced(this);
    findingRecord<IdealGeometryRecord>();
}

XMLIdealGeometryESSource::~XMLIdealGeometryESSource() {}

const DDCompactView *
XMLIdealGeometryESSource::produce(const IdealGeometryRecord &)
{ return new DDCompactView(); }

void XMLIdealGeometryESSource::setIntervalFor(const EventSetupRecordKey &,
					       const edm::Timestamp &, 
					       edm::ValidityInterval & oValidity)
{
   edm::ValidityInterval infinity(edm::Timestamp(1), edm::Timestamp::endOfTime());
   oValidity = infinity;
}


#include "FWCore/Framework/interface/SourceFactory.h"


DEFINE_FWK_EVENTSETUP_SOURCE(XMLIdealGeometryESSource)


