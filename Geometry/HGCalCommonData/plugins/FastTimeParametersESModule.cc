#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/FastTimeParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/FastTimeParametersFromDD.h"

#include <memory>

//#define EDM_ML_DEBUG

class  FastTimeParametersESModule : public edm::ESProducer {
public:
  FastTimeParametersESModule(const edm::ParameterSet &);
  ~FastTimeParametersESModule(void) override;
  
  using ReturnType = std::unique_ptr<FastTimeParameters>;

  static void fillDescriptions( edm::ConfigurationDescriptions & ) { }
  
  ReturnType produce(const IdealGeometryRecord&);

private:
  std::vector<std::string> name_;
  std::vector<int>         type_;
};

FastTimeParametersESModule::FastTimeParametersESModule(const edm::ParameterSet& iC) {

  name_  = iC.getUntrackedParameter<std::vector<std::string> >("Names");
  type_  = iC.getUntrackedParameter<std::vector<int> >("Types");
  edm::LogInfo("HGCalGeom") << "FastTimeParametersESModule for " 
			    << name_.size() << " types";
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeParametersESModule for " << name_.size() << " types:";
  for (unsigned int k=0; k<name_.size(); ++k)
    std::cout << " [" << k << "] " << name_[k] << ":" << type_[k];
  std::cout << std::endl;
#endif
  setWhatProduced(this);
}

FastTimeParametersESModule::~FastTimeParametersESModule() {}

FastTimeParametersESModule::ReturnType
FastTimeParametersESModule::produce(const IdealGeometryRecord& iRecord) {
  edm::LogInfo("HGCalGeom")
    <<  "FastTimeParametersESModule::produce(const IdealGeometryRecord& iRecord)";
  edm::ESTransientHandle<DDCompactView> cpv;
  iRecord.get(cpv);
  
  auto ptp = std::make_unique<FastTimeParameters>();
  FastTimeParametersFromDD builder;
  for  (unsigned int k=0; k<name_.size(); ++k)
    builder.build(&(*cpv), *ptp, name_[k], type_[k]);
  
#ifdef EDM_ML_DEBUG
  std::cout << "FastTimeParametersESModule:: Barrel Parameters: " 
	    << " number of cells along z|phi = " << ptp->nZBarrel_ << "|"
	    << ptp->nPhiBarrel_ << " Geometry parameters = ( ";
  for (unsigned k=0; k<ptp->geomParBarrel_.size(); ++k)
    std::cout << "[" << k << "] " << ptp->geomParBarrel_[k] << " ";
  std::cout << ")" << std::endl;
  std::cout << "FastTimeParametersESModule:: Endcap Parameters: " 
	    << " number of cells along eta|phi = " << ptp->nEtaEndcap_ << "|"
	    << ptp->nPhiEndcap_ << " Geometry parameters = ( ";
  for (unsigned k=0; k<ptp->geomParEndcap_.size(); ++k)
    std::cout << "[" << k << "] " << ptp->geomParEndcap_[k] << " ";
  std::cout << ")" << std::endl;
#endif
  return ptp;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(FastTimeParametersESModule);
