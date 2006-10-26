#include "GeometryReaders/XMLIdealGeometryESSource/interface/XMLIdealGeometryESSource.h"
#include "GeometryReaders/XMLIdealGeometryESSource/interface/GeometryConfiguration.h"

#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "SealUtil/SealTimer.h"

#include <memory>


XMLIdealGeometryESSource::XMLIdealGeometryESSource(const edm::ParameterSet & p): rootNodeName_(p.getParameter<std::string>("rootNodeName"))
{
    DDLParser * parser = DDLParser::instance();
    GeometryConfiguration cf(p);
//     edm::FileInPath fp = p.getParameter<edm::FileInPath>("GeometryConfiguration");
//     DCOUT ('X', "FileInPath is looking for " + fp.fullPath());
//     int result1 = cf.readConfig(fp.fullPath());
//     if (result1 !=0) throw DDException("DDLConfiguration: readConfig failed !");
//    if(rootNodeName_ == "" || rootNodeName_ == "\""){
//       rootNodeName_ = DDRootDef::instance().root().ddname();
//    }
    if ( rootNodeName_ == "" || rootNodeName_ == "\\" ) {
      throw DDException ("XMLIdealGeometryESSource must have a root node name.");
    }

    DDRootDef::instance().set(DDName(rootNodeName_));
    //    std::cout << "SET ROOT NODE TO: " << rootNodeName_ << std::endl;
    seal::SealTimer txml("XMLIdealGeometryESSource");

    int result2 = parser->parse(cf);

    if (result2 != 0) throw DDException("DDD-Parser: parsing failed!");

    if ( !bool(DDLogicalPart( DDName(rootNodeName_) )) ) {
      throw DDException ("XMLIdealGeometryESSource was given a non-existent node name for the root. " + rootNodeName_ );
    }

    //use the label specified in the configuration file as the 
    // label client code must use to get the DDCompactView
    if(""==p.getParameter<std::string>("@module_label")){
       setWhatProduced(this);
    }else {
       setWhatProduced(this,p.getParameter<std::string>("@module_label"));
    }
    findingRecord<IdealGeometryRecord>();
}

XMLIdealGeometryESSource::~XMLIdealGeometryESSource() {}

std::auto_ptr<DDCompactView>
XMLIdealGeometryESSource::produce(const IdealGeometryRecord &)
{ 
   //std::cout <<"got in produce"<<std::endl;
   DDName ddName(rootNodeName_);
   //std::cout <<"ddName \""<<ddName<<"\""<<std::endl;
   DDLogicalPart rootNode(ddName);
   //std::cout <<"made the DDLogicalPart"<<std::endl;
   if(! rootNode.isValid()){
      throw cms::Exception("Geometry")<<"There is no valid node named \""
				      <<rootNodeName_<<"\"";
   }
   std::auto_ptr<DDCompactView> returnValue(new DDCompactView(rootNode));
   //copy the graph from the global one
   DDCompactView globalOne;
   returnValue->writeableGraph() = globalOne.graph();
   //std::cout <<"made the view"<<std::endl;
   return returnValue;
}

void XMLIdealGeometryESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
					       const edm::IOVSyncValue & iosv, 
					       edm::ValidityInterval & oValidity)
{
   edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
   oValidity = infinity;
}


#include "FWCore/Framework/interface/SourceFactory.h"


DEFINE_FWK_EVENTSETUP_SOURCE(XMLIdealGeometryESSource);


