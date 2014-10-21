#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"

#include "DetectorDescription/Core/src/Material.h"
#include "DetectorDescription/Core/src/Solid.h"
#include "DetectorDescription/Core/src/LogicalPart.h"
#include "DetectorDescription/Core/src/Specific.h"

#include <memory>

class XMLIdealMagneticFieldGeometryESProducer : public edm::ESProducer
{
public:
  XMLIdealMagneticFieldGeometryESProducer( const edm::ParameterSet& );
  ~XMLIdealMagneticFieldGeometryESProducer();
  
  typedef std::auto_ptr<DDCompactView> ReturnType;
  
  ReturnType produce( const IdealMagneticFieldRecord& );

private:
  std::string rootDDName_; // this must be the form namespace:name
  std::string label_;

  DDI::Store<DDName, DDI::Material*> matStore_;
  DDI::Store<DDName, DDI::Solid*> solidStore_;
  DDI::Store<DDName, DDI::LogicalPart*> lpStore_;
  DDI::Store<DDName, DDI::Specific*> specStore_;
  DDI::Store<DDName, DDRotationMatrix*> rotStore_;    
};

XMLIdealMagneticFieldGeometryESProducer::XMLIdealMagneticFieldGeometryESProducer( const edm::ParameterSet& iConfig )
  :   rootDDName_(iConfig.getParameter<std::string>( "rootDDName" )),
      label_(iConfig.getParameter<std::string>( "label" ))
{
  setWhatProduced( this );
}


XMLIdealMagneticFieldGeometryESProducer::~XMLIdealMagneticFieldGeometryESProducer( void )
{}

XMLIdealMagneticFieldGeometryESProducer::ReturnType
XMLIdealMagneticFieldGeometryESProducer::produce( const IdealMagneticFieldRecord& iRecord )
{
  using namespace edm::es;

  edm::ESTransientHandle<FileBlob> gdd;
  iRecord.getRecord<GeometryFileRcd>().get( label_, gdd );

  DDName ddName(rootDDName_);
  DDLogicalPart rootNode(ddName);
  DDRootDef::instance().set(rootNode);
  ReturnType returnValue(new DDCompactView(rootNode));
  DDLParser parser(*returnValue);
  parser.getDDLSAX2FileHandler()->setUserNS(true);
  parser.clearFiles();

  std::unique_ptr<std::vector<unsigned char> > tb = (*gdd).getUncompressedBlob();

  parser.parse(*tb, tb->size());

  returnValue->lockdown();

  return returnValue ;
}

DEFINE_FWK_EVENTSETUP_MODULE( XMLIdealMagneticFieldGeometryESProducer );
