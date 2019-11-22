#include "CondTools/SiPixel/test/SiPixelTemplateDBObjectUploader.h"
#include "CondFormats/DataRecord/interface/SiPixelTemplateDBObjectRcd.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <fstream>

SiPixelTemplateDBObjectUploader::SiPixelTemplateDBObjectUploader(const edm::ParameterSet& iConfig)
    : theTemplateCalibrations(iConfig.getParameter<vstring>("siPixelTemplateCalibrations")),
      theTemplateBaseString(iConfig.getParameter<std::string>("theTemplateBaseString")),
      theVersion(iConfig.getParameter<double>("Version")),
      theMagField(iConfig.getParameter<double>("MagField")),
      theDetIds(iConfig.getParameter<std::vector<uint32_t> >("detIds")),
      theTemplIds(iConfig.getParameter<std::vector<uint32_t> >("templateIds")) {}

SiPixelTemplateDBObjectUploader::~SiPixelTemplateDBObjectUploader() {}

void SiPixelTemplateDBObjectUploader::beginJob() {}

void SiPixelTemplateDBObjectUploader::analyze(const edm::Event& iEvent, const edm::EventSetup& es) {
  //--- Make the POOL-ORA object to store the database object
  SiPixelTemplateDBObject* obj = new SiPixelTemplateDBObject;

  // Local variables
  const char* tempfile;
  int m;

  // Set the number of templates to be passed to the dbobject
  obj->setNumOfTempl(theTemplateCalibrations.size());

  // Set the version of the template dbobject - this is an external parameter
  obj->setVersion(theVersion);

  // Open the template file(s)
  for (m = 0; m < obj->numOfTempl(); ++m) {
    edm::FileInPath file(theTemplateCalibrations[m].c_str());
    tempfile = (file.fullPath()).c_str();

    std::ifstream in_file(tempfile, std::ios::in);

    if (in_file.is_open()) {
      edm::LogInfo("Template Info") << "Opened Template File: " << file.fullPath().c_str();

      // Local variables
      char title_char[80], c;
      SiPixelTemplateDBObject::char2float temp;
      float tempstore;
      int iter, j;

      // Templates contain a header char - we must be clever about storing this
      for (iter = 0; (c = in_file.get()) != '\n'; ++iter) {
        if (iter < 79) {
          title_char[iter] = c;
        }
      }
      if (iter > 78) {
        iter = 78;
      }
      title_char[iter + 1] = '\n';

      for (j = 0; j < 80; j += 4) {
        temp.c[0] = title_char[j];
        temp.c[1] = title_char[j + 1];
        temp.c[2] = title_char[j + 2];
        temp.c[3] = title_char[j + 3];
        obj->push_back(temp.f);
        obj->setMaxIndex(obj->maxIndex() + 1);
      }

      // Fill the dbobject
      in_file >> tempstore;
      while (!in_file.eof()) {
        obj->setMaxIndex(obj->maxIndex() + 1);
        obj->push_back(tempstore);
        in_file >> tempstore;
      }

      in_file.close();
    } else {
      // If file didn't open, report this
      edm::LogError("SiPixelTemplateDBObjectUploader") << "Error opening File" << tempfile;
    }
  }

  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord>().get(pDD);

  for (unsigned int i = 0; i < theDetIds.size(); ++i) {
    short s_detid = (short)theDetIds[i];
    short templid = (short)theTemplIds[i];

    DetId theDetid(s_detid);
    if (s_detid != 0 && s_detid != 1 && s_detid != 2) {
      if (!(*obj).putTemplateID(theDetid.rawId(), templid)) {
        edm::LogInfo("Template Info") << " Could not fill specified det unit: " << theDetid;
      }
    } else {
      edm::LogInfo("DetUnit Info") << " There are " << pDD->detUnits().size() << " detectors";
    }
    for (const auto& it : pDD->detUnits()) {
      if (dynamic_cast<PixelGeomDetUnit const*>(it) != 0) {
        DetId detid = it->geographicalId();

        if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel) &&
            (detid.subdetId() == s_detid || s_detid == 0)) {
          if (!(*obj).putTemplateID(detid.rawId(), templid))
            edm::LogInfo("Template Info") << " Could not fill barrel det unit";
        }
        if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap) &&
            (detid.subdetId() == s_detid || s_detid == 0)) {
          if (!(*obj).putTemplateID(detid.rawId(), templid))
            edm::LogInfo("Template Info") << " Could not fill endcap det unit";
        }
      } else {
        //edm::LogInfo("Template Info")<< "Detid is Pixel but neither bpix nor fpix";
      }
    }
  }

  // Uncomment to output the contents of the db object at the end of the job
  //	std::cout << *obj << std::endl;
  //std::map<unsigned int,short> templMap=(*obj).getTemplateIDs();
  //for(std::map<unsigned int,short>::const_iterator it=templMap.begin(); it!=templMap.end();++it)
  //std::cout<< "Map:\n"<< "DetId: "<< it->first << " TemplateID: "<< it->second <<"\n";

  //--- Create a new IOV
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())  // Die if not available
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  if (poolDbService->isNewTagRequest("SiPixelTemplateDBObjectRcd"))
    poolDbService->writeOne(obj, poolDbService->beginOfTime(), "SiPixelTemplateDBObjectRcd");
  else
    poolDbService->writeOne(obj, poolDbService->currentTime(), "SiPixelTemplateDBObjectRcd");
}

void SiPixelTemplateDBObjectUploader::endJob() {}
