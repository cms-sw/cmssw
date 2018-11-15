#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "CondFormats/Common/interface/FileBlob.h"

#include <memory>

class XMLIdealGeometryESProducer : public edm::ESProducer
{
public:
  XMLIdealGeometryESProducer(const edm::ParameterSet&);
  
  using ReturnType = std::unique_ptr<DDCompactView>;
  
  ReturnType produce(const IdealGeometryRecord&);

private:

  std::string rootDDName_; // this must be the form namespace:name
  std::string label_;
};

XMLIdealGeometryESProducer::XMLIdealGeometryESProducer( const edm::ParameterSet& iConfig )
  : rootDDName_( iConfig.getParameter<std::string>( "rootDDName" )),
    label_( iConfig.getParameter<std::string>( "label" ))
{
  setWhatProduced( this );
}

XMLIdealGeometryESProducer::ReturnType
XMLIdealGeometryESProducer::produce( const IdealGeometryRecord& iRecord )
{
  edm::ESTransientHandle<FileBlob> gdd;
  iRecord.getRecord<GeometryFileRcd>().get( label_, gdd );
  auto cpv = std::make_unique<DDCompactView>( DDName( rootDDName_ ));
  DDLParser parser( *cpv );
  parser.getDDLSAX2FileHandler()->setUserNS( true );
  parser.clearFiles();
  
  std::unique_ptr<std::vector<unsigned char> > tb = (*gdd).getUncompressedBlob();
  
  parser.parse( *tb, tb->size());
  
  cpv->lockdown();
  
  return cpv;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(XMLIdealGeometryESProducer);
