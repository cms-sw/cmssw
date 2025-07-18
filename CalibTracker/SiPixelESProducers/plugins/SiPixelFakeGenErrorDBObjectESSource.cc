// system includes
#include <memory>
#include <fstream>

// user includes
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class SiPixelFakeGenErrorDBObjectESSource : public edm::ESProducer, public edm::EventSetupRecordIntervalFinder {
public:
  SiPixelFakeGenErrorDBObjectESSource(const edm::ParameterSet&);
  ~SiPixelFakeGenErrorDBObjectESSource() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  typedef std::vector<std::string> vstring;

  virtual std::unique_ptr<SiPixelGenErrorDBObject> produce(const SiPixelGenErrorDBObjectRcd&);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                      const edm::IOVSyncValue&,
                      edm::ValidityInterval&) override;

private:
  vstring GenErrorCalibrations_;
  float version_;
};

SiPixelFakeGenErrorDBObjectESSource::SiPixelFakeGenErrorDBObjectESSource(const edm::ParameterSet& conf_)
    : GenErrorCalibrations_(conf_.getParameter<vstring>("siPixelGenErrorCalibrations")),
      version_(conf_.getParameter<double>("Version")) {
  edm::LogInfo("SiPixelFakeGenErrorDBObjectESSource::SiPixelFakeGenErrorDBObjectESSource");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelGenErrorDBObjectRcd>();
}

std::unique_ptr<SiPixelGenErrorDBObject> SiPixelFakeGenErrorDBObjectESSource::produce(
    const SiPixelGenErrorDBObjectRcd&) {
  using namespace edm::es;

  //Mostly copied from CondTools/SiPixel/test/SiPixelGenErrorDBObjectUploader.cc
  //--- Make the POOL-ORA object to store the database object
  SiPixelGenErrorDBObject* obj = new SiPixelGenErrorDBObject;

  // Local variables
  int m;

  // Set the number of GenErrors to be passed to the dbobject
  obj->setNumOfTempl(GenErrorCalibrations_.size());

  // Set the version of the GenError dbobject - this is an external parameter
  obj->setVersion(version_);

  //  open the GenError file(s)
  for (m = 0; m < obj->numOfTempl(); ++m) {
    edm::FileInPath file(GenErrorCalibrations_[m].c_str());
    std::ifstream in_file(file.fullPath(), std::ios::in);

    if (in_file.is_open()) {
      edm::LogPrint("SiPixelFakeGenErrorDBObjectESSource")
          << "Opened GenError File: " << file.fullPath().c_str() << std::endl;

      // Local variables
      char title_char[80], c;
      SiPixelGenErrorDBObject::char2float temp;
      float tempstore;
      unsigned int iter;

      // GenErrors contain a header char - we must be clever about storing this
      for (iter = 0; (c = in_file.get()) != '\n' && iter < 79; ++iter) {
        title_char[iter] = c;
      }
      if (iter == 79) {
        title_char[iter] = '\n';
      } else {
        unsigned int ilast = 3 - (iter % 4);
        for (unsigned int it = 0; it != ilast; it++) {
          title_char[iter] = ' ';
          iter++;
        }
        title_char[iter] = '\n';
      }

      for (unsigned int j = 0; j <= iter; j += 4) {
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
      edm::LogError("SiPixeFakelGenErrorDBObjectESSource") << "Error opening File" << file.fullPath() << std::endl;
    }
  }

  //std::cout << *obj << std::endl;
  return std::unique_ptr<SiPixelGenErrorDBObject>(obj);
}

void SiPixelFakeGenErrorDBObjectESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                         const edm::IOVSyncValue& iosv,
                                                         edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

void SiPixelFakeGenErrorDBObjectESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<vstring>(
      "siPixelGenErrorCalibrations",
      {"CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1/generror_summary_zp0310.out",
       "CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1/generror_summary_zp0311.out",
       "CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1/generror_summary_zp0312.out",
       "CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1/generror_summary_zp0313.out",
       "CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1/generror_summary_zp0314.out",
       "CalibTracker/SiPixelESProducers/data/SiPixelTemplateDBObject_0T_phase1_BoR3_v1/generror_summary_zp0315.out"});
  desc.add<double>("Version", 1.0);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_EVENTSETUP_SOURCE(SiPixelFakeGenErrorDBObjectESSource);
