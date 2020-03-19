#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/HGCalCommonData/interface/FastTimeParameters.h"
#include "Geometry/HGCalCommonData/interface/FastTimeParametersFromDD.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

//#define EDM_ML_DEBUG

class FastTimeParametersESModule : public edm::ESProducer {
public:
  FastTimeParametersESModule(const edm::ParameterSet&);
  ~FastTimeParametersESModule(void) override;

  using ReturnType = std::unique_ptr<FastTimeParameters>;

  static void fillDescriptions(edm::ConfigurationDescriptions&) {}

  ReturnType produce(const IdealGeometryRecord&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> cpvToken_;
  std::vector<std::string> name_;
  std::vector<int> type_;
};

FastTimeParametersESModule::FastTimeParametersESModule(const edm::ParameterSet& iC)
    : cpvToken_{setWhatProduced(this).consumes<DDCompactView>(edm::ESInputTag{})} {
  name_ = iC.getUntrackedParameter<std::vector<std::string> >("Names");
  type_ = iC.getUntrackedParameter<std::vector<int> >("Types");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "FastTimeParametersESModule for " << name_.size() << " types:";
  for (unsigned int k = 0; k < name_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << " [" << k << "] " << name_[k] << ":" << type_[k];
#endif
}

FastTimeParametersESModule::~FastTimeParametersESModule() {}

FastTimeParametersESModule::ReturnType FastTimeParametersESModule::produce(const IdealGeometryRecord& iRecord) {
  edm::LogVerbatim("HGCalGeom") << "FastTimeParametersESModule::produce(const IdealGeometryRecord& iRecord)";
  edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvToken_);

  auto ptp = std::make_unique<FastTimeParameters>();
  FastTimeParametersFromDD builder;
  for (unsigned int k = 0; k < name_.size(); ++k)
    builder.build(cpv.product(), *ptp, name_[k], type_[k]);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "FastTimeParametersESModule:: Barrel Parameters: "
                                << " number of cells along z|phi = " << ptp->nZBarrel_ << "|" << ptp->nPhiBarrel_
                                << " Geometry parameters:";
  for (unsigned k = 0; k < ptp->geomParBarrel_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << ptp->geomParBarrel_[k];
  edm::LogVerbatim("HGCalGeom") << "FastTimeParametersESModule:: Endcap Parameters: "
                                << " number of cells along eta|phi = " << ptp->nEtaEndcap_ << "|" << ptp->nPhiEndcap_
                                << " Geometry parameters:";
  for (unsigned k = 0; k < ptp->geomParEndcap_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << ptp->geomParEndcap_[k];
#endif
  return ptp;
}

// define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(FastTimeParametersESModule);
