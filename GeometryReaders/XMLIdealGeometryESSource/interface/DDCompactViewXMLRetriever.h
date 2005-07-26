#ifndef SimG4Core_DDCompactViewXMLRetriever_H
#define SimG4Core_DDCompactViewXMLRetriever_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DetectorDescription/DDCore/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <string>

class DDCompactViewXMLRetriever : public edm::eventsetup::ESProducer, 
                                  public edm::eventsetup::EventSetupRecordIntervalFinder
{
public:
    DDCompactViewXMLRetriever(const edm::ParameterSet & p);
    virtual ~DDCompactViewXMLRetriever(); 
    const DDCompactView * produce(const IdealGeometryRecord &);
protected:
    virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
				const edm::Timestamp &,edm::ValidityInterval &);
private:
    DDCompactViewXMLRetriever(const DDCompactViewXMLRetriever &);
    const DDCompactViewXMLRetriever & operator=(const DDCompactViewXMLRetriever &);
};


#endif
