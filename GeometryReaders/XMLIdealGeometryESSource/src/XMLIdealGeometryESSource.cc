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

#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "DetectorDescription/Core/src/Specific.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"

#include <memory>


XMLIdealGeometryESSource::XMLIdealGeometryESSource(const edm::ParameterSet & p): rootNodeName_(p.getParameter<std::string>("rootNodeName")),
                                                                                 userNS_(p.getUntrackedParameter<bool>("userControlledNamespace", false)),
                                                                                 geoConfig_(p)
{
  if ( rootNodeName_ == "" || rootNodeName_ == "\\" ) {
    throw DDException ("XMLIdealGeometryESSource must have a root node name.");
  }
  
  if ( rootNodeName_ == "MagneticFieldVolumes:MAGF" ||  rootNodeName_ == "cmsMagneticField:MAGF") {
    setWhatProduced(this, &XMLIdealGeometryESSource::produceMagField, 
                    edm::es::Label(p.getParameter<std::string>("@module_label")));
    findingRecord<IdealMagneticFieldRecord>();
  } else {
    setWhatProduced(this, &XMLIdealGeometryESSource::produceGeom, 
                    edm::es::Label(p.getParameter<std::string>("@module_label")));
    findingRecord<IdealGeometryRecord>();
  }
}

XMLIdealGeometryESSource::~XMLIdealGeometryESSource() { }

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
  
  // 2009-07-09 memory patch

  // unlock before use because it can be used more than once!
  DDMaterial::StoreT::instance().setReadOnly(false);
  DDSolid::StoreT::instance().setReadOnly(false);
  DDLogicalPart::StoreT::instance().setReadOnly(false);
  DDSpecifics::StoreT::instance().setReadOnly(false);
  DDRotation::StoreT::instance().setReadOnly(false);

  DDName ddName(rootNodeName_);
  DDLogicalPart rootNode(ddName);
  DDRootDef::instance().set(rootNode);
  std::auto_ptr<DDCompactView> returnValue(new DDCompactView(rootNode));
  DDLParser parser(*returnValue); //* parser = DDLParser::instance();
  parser.getDDLSAX2FileHandler()->setUserNS(userNS_);
  int result2 = parser.parse(geoConfig_);
  if (result2 != 0) throw DDException("DDD-Parser: parsing failed!");

  // after parsing the root node should be valid!

  if( !rootNode.isValid() ){
    throw cms::Exception("Geometry")<<"There is no valid node named \""
                                    <<rootNodeName_<<"\"";
  }

  // at this point we should have a valid store of DDObjects and we will move these
  // to the local storage area using swaps with the existing Singleton<Store...>'s
  // 2009-07-09 memory patch
  DDMaterial::StoreT::instance().swap(matStore_);
  DDSolid::StoreT::instance().swap(solidStore_);
  DDLogicalPart::StoreT::instance().swap(lpStore_);
  DDSpecifics::StoreT::instance().swap(specStore_);
  DDRotation::StoreT::instance().swap(rotStore_);

  // lock the global stores.
  DDMaterial::StoreT::instance().setReadOnly(false);
  DDSolid::StoreT::instance().setReadOnly(false);
  DDLogicalPart::StoreT::instance().setReadOnly(false);
  DDSpecifics::StoreT::instance().setReadOnly(false);
  DDRotation::StoreT::instance().setReadOnly(false);
  
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


