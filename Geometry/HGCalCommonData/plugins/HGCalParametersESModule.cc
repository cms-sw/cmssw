#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalParametersFromDD.h"

#include <memory>

//#define DeugLog

class  HGCalParametersESModule : public edm::ESProducer {
public:
  HGCalParametersESModule( const edm::ParameterSet & );
  ~HGCalParametersESModule( void );
  
  typedef std::shared_ptr<HGCalParameters> ReturnType;
  
  ReturnType produce( const IdealGeometryRecord&);

private:
  std::string        name_, namew_, namec_;
};

HGCalParametersESModule::HGCalParametersESModule(const edm::ParameterSet& iC) {

  name_  = iC.getUntrackedParameter<std::string>("Name");
  namew_ = iC.getUntrackedParameter<std::string>("NameW");
  namec_ = iC.getUntrackedParameter<std::string>("NameC");
  edm::LogInfo("HGCalGeom") << "HGCalParametersESModule for " << name_ << ":"
			    << namew_ << ":" << namec_;
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalParametersESModule for " << name_ << ":" << namew_ << ":" 
	    << namec_ << std::endl;
#endif
  setWhatProduced(this, name_);
}

HGCalParametersESModule::~HGCalParametersESModule() {}

HGCalParametersESModule::ReturnType
HGCalParametersESModule::produce(const IdealGeometryRecord& iRecord) {
  edm::LogInfo("HGCalGeom")
    <<  "HGCalParametersESModule::produce(const IdealGeometryRecord& iRecord)";
  edm::ESTransientHandle<DDCompactView> cpv;
  iRecord.get(cpv);
  
  HGCalParameters* ptp = new HGCalParameters(name_);
  HGCalParametersFromDD builder;
  builder.build(&(*cpv), *ptp, name_, namew_, namec_);
  
  return ReturnType(ptp) ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(HGCalParametersESModule);
