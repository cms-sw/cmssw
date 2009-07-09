#include "GeometryReaders/XMLIdealGeometryESSource/interface/XMLIdealGeometryESSource.h"

#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"

#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>


XMLIdealGeometryESSource::XMLIdealGeometryESSource(const edm::ParameterSet & p): rootNodeName_(p.getParameter<std::string>("rootNodeName")),
                                                                                 userNS_(p.getUntrackedParameter<bool>("userControlledNamespace", false)),
                                                                                 cpvavailable_(false),geoConfig_(p)
{
  if ( rootNodeName_ == "" || rootNodeName_ == "\\" ) {
    throw DDException ("XMLIdealGeometryESSource must have a root node name.");
  }
  
  if ( rootNodeName_ == "MagneticFieldVolumes:MAGF" ||  rootNodeName_ == "cmsMagneticField:MAGF") {
    setWhatProduced(this, &XMLIdealGeometryESSource::produceMagField, 
                    edm::es::Label(p.getParameter<std::string>("@module_label")));
    findingRecord<IdealMagneticFieldRecord>();
    std::cout <<"finding Mag field" << std::endl;
  } else {
    setWhatProduced(this, &XMLIdealGeometryESSource::produceGeom);
    findingRecord<IdealGeometryRecord>();
    std::cout << "finding ideal geom" << std::endl;
  }
}

XMLIdealGeometryESSource::~XMLIdealGeometryESSource() {
  if(cpvavailable_){
    DDCompactView cpv;
    cpv.clear();
  }
}

std::auto_ptr<DDCompactView>
XMLIdealGeometryESSource::produceGeom(const IdealGeometryRecord &)
{
  return produce();
}

std::auto_ptr<DDCompactView>
XMLIdealGeometryESSource::produceMagField(const IdealMagneticFieldRecord &)
{ 
  return produce();
}


std::auto_ptr<DDCompactView>
XMLIdealGeometryESSource::produce() {
  
  cpvavailable_ = true;

  DDLParser * parser = DDLParser::instance();
  // 2009-07-09 memory patch
  parser->clearFiles();

  parser->getDDLSAX2FileHandler()->setUserNS(userNS_);
  DDRootDef::instance().set(DDName(rootNodeName_));
  
  int result2 = parser->parse(geoConfig_);
  const std::vector<std::string> & whatsparsed = geoConfig_.getFileList();
  for (std::vector<std::string>::const_iterator it = whatsparsed.begin(); it != whatsparsed.end(); ++it ) {
    std::cout << *it << std::endl;
  }  
  if (result2 != 0) throw DDException("DDD-Parser: parsing failed!");

  if ( !bool(DDLogicalPart( DDName(rootNodeName_) )) ) {
    throw DDException ("XMLIdealGeometryESSource was given a non-existent node name for the root. " + rootNodeName_ );
  }

  //std::cout <<"got in produce"<<std::endl;
  DDName ddName(rootNodeName_);
  //std::cout <<"ddName \""<<ddName<<"\""<<std::endl;
  DDLogicalPart rootNode(ddName);
  //std::cout <<"made the DDLogicalPart"<<std::endl;
  if(! rootNode.isValid()){
    throw cms::Exception("Geometry")<<"There is no valid node named \""
                                    <<rootNodeName_<<"\"";
  }

  // at this point we should have a valid store of DDObjects and we will move these
  // to the local storage area using swaps with the existing Singleton<Store...>'s
  // NOTE TO SELF:  this is similar to the below explicit copy of the graph of the
  // DDCompactView at this point with this XMLIdealGeometryESSource so as not to
  // share the Store's anymore.
  // 2009-07-09 memory patch
  std::cout << "before swap size = " << DDMaterial::StoreT::instance().size() << std::endl;
   DDMaterial::StoreT::instance().swap(matStore_);
   std::cout << "after swap size = " << DDMaterial::StoreT::instance().size() << std::endl;
   std::cout << "and matStore_.size() = " << matStore_.size() << std::endl; 
   DDSolid::StoreT::instance().swap(solidStore_);
   DDLogicalPart::StoreT::instance().swap(lpStore_);
   DDSpecifics::StoreT::instance().swap(specStore_);
   DDRotation::StoreT::instance().swap(rotStore_);

  std::auto_ptr<DDCompactView> returnValue(new DDCompactView(rootNode));
  
  // NOTE TO SELF:  Mike, DO NOT try to fix the memory leak here by going global again!!!
  //copy the graph from the global one
  DDCompactView globalOne;
  returnValue->writeableGraph() = globalOne.graph();
  
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


