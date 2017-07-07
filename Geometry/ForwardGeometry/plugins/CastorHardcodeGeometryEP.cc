#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "Geometry/ForwardGeometry/plugins/CastorHardcodeGeometryEP.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


CastorHardcodeGeometryEP::CastorHardcodeGeometryEP(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced( this, CastorGeometry::producerTag() );

   loader_=nullptr;
}


CastorHardcodeGeometryEP::~CastorHardcodeGeometryEP()
{ 
  if (loader_) delete loader_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CastorHardcodeGeometryEP::ReturnType
CastorHardcodeGeometryEP::produce(const CastorGeometryRecord& iRecord)
{
   loader_=new CastorHardcodeGeometryLoader();
   return std::unique_ptr<CaloSubdetectorGeometry>(loader_->load());
}


