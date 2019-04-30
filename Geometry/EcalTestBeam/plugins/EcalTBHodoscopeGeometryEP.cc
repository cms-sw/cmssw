// -*- C++ -*-
//
// Package:    EcalTBHodoscopeGeometryEP
// Class:      EcalTBHodoscopeGeometryEP
// 
/**\class EcalTBHodoscopeGeometryEP    

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//

//
//

#include "Geometry/EcalTestBeam/plugins/EcalTBHodoscopeGeometryEP.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
EcalTBHodoscopeGeometryEP::EcalTBHodoscopeGeometryEP(const edm::ParameterSet& iConfig):
  cpvToken_{setWhatProduced(this,"EcalLaserPnDiode").consumes<DDCompactView>(edm::ESInputTag{})}
{}


//
// member functions
//

// ------------ method called to produce the data  ------------
EcalTBHodoscopeGeometryEP::ReturnType
EcalTBHodoscopeGeometryEP::produce(const IdealGeometryRecord& iRecord)
{

   edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(cpvToken_);
   
   LogDebug("EcalTBHodoscopeGeometryEP") << "[EcalTBHodoscopeGeometryEP]::Constructing EcalTBHodoscopeGeometry";
   return std::unique_ptr<CaloSubdetectorGeometry>(loader_.load(&(*cpv)));
}


