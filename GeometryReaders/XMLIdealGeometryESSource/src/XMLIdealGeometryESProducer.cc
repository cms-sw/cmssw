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

class XMLIdealGeometryESProducer : public edm::ESProducer {
public:
  XMLIdealGeometryESProducer(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<DDCompactView>;

  ReturnType produce(const IdealGeometryRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const std::string rootDDName_;  // this must be the form namespace:name
  edm::ESGetToken<FileBlob, GeometryFileRcd> blobToken_;
};

XMLIdealGeometryESProducer::XMLIdealGeometryESProducer(const edm::ParameterSet& iConfig)
    : rootDDName_(iConfig.getParameter<std::string>("rootDDName")) {
  setWhatProduced(this).setConsumes(blobToken_, edm::ESInputTag("", iConfig.getParameter<std::string>("label")));
}

XMLIdealGeometryESProducer::ReturnType XMLIdealGeometryESProducer::produce(const IdealGeometryRecord& iRecord) {
  edm::ESTransientHandle<FileBlob> gdd = iRecord.getTransientHandle(blobToken_);
  auto cpv = std::make_unique<DDCompactView>(DDName(rootDDName_));
  DDLParser parser(*cpv);
  parser.getDDLSAX2FileHandler()->setUserNS(true);
  parser.clearFiles();

  std::unique_ptr<std::vector<unsigned char> > tb = (*gdd).getUncompressedBlob();

  parser.parse(*tb, tb->size());

  cpv->lockdown();

  return cpv;
}

void XMLIdealGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("rootDDName")->setComment("The value must be of the form 'namespace:name'");
  desc.add<std::string>("label")->setComment("product label used to get the FileBlob");

  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(XMLIdealGeometryESProducer);
