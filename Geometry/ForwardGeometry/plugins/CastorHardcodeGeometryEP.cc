#include "Geometry/ForwardGeometry/plugins/CastorHardcodeGeometryEP.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


CastorHardcodeGeometryEP::CastorHardcodeGeometryEP(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this,"CASTOR");
   loader_=0;
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
  //using namespace edm::es;
  if (loader_==0) {
    edm::ESHandle<CastorTopology> topo;
    try {
      iRecord.get(topo);
      loader_=new CastorHardcodeGeometryLoader(*topo); 
    } catch (...) {
      loader_=new CastorHardcodeGeometryLoader();
       edm::LogInfo("CASTOR") << "Using default Castor topology";
         }
        }
   std::auto_ptr<CaloSubdetectorGeometry> pCaloSubdetectorGeometry(loader_->load()) ;

   return pCaloSubdetectorGeometry ;
}


