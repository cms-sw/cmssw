// -*- C++ -*-
//
// Package:    ZdcHardcodeGeometryEP
// Class:      ZdcHardcodeGeometryEP
// 
/**\class ZdcHardcodeGeometryEP ZdcHardcodeGeometryEP.h
   
    Description: <one line class summary>

    Implementation:
    <Notes on implementation>
*/
//
// Original Author:  Edmundo Garcia
//         Created:  Mon Aug  6 12:33:33 CDT 2007
// $Id: ZdcHardcodeGeometryEP.cc,v 1.0 2007/08/06 14:16:12 ejgarcia Exp $
//
#include "Geometry/ForwardGeometry/plugins/ZdcHardcodeGeometryEP.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


ZdcHardcodeGeometryEP::ZdcHardcodeGeometryEP(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this,"ZDC");
   loader_=0;
}


ZdcHardcodeGeometryEP::~ZdcHardcodeGeometryEP()
{ 
  if (loader_) delete loader_;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
ZdcHardcodeGeometryEP::ReturnType
ZdcHardcodeGeometryEP::produce(const IdealGeometryRecord& iRecord)
{
  //using namespace edm::es;
  if (loader_==0) {
    edm::ESHandle<ZdcTopology> topo;
    try {
      iRecord.get(topo);
      loader_=new ZdcHardcodeGeometryLoader(*topo); 
    } catch (...) {
      loader_=new ZdcHardcodeGeometryLoader();
       edm::LogInfo("ZDC") << "Using default ZDC topology";
         }
        }
   std::auto_ptr<CaloSubdetectorGeometry> pCaloSubdetectorGeometry(loader_->load()) ;

   return pCaloSubdetectorGeometry ;
}


