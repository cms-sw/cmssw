// -*- C++ -*-
//
// Package:    CaloGeometryBuilder
// Class:      CaloGeometryBuilder
// 
/**\class CaloGeometryBuilder CaloGeometryBuilder.h tmp/CaloGeometryBuilder/interface/CaloGeometryBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

//
// class decleration
//

class CaloGeometryBuilder : public edm::ESProducer 
{
   public:

      using ReturnType = std::unique_ptr<CaloGeometry>;

      typedef edm::ESHandle<CaloSubdetectorGeometry> SubdType ;

      CaloGeometryBuilder( const edm::ParameterSet& iConfig ) ;

      ~CaloGeometryBuilder() override {} ;

      ReturnType produceAligned( const CaloGeometryRecord&  iRecord ) ;

   private:
      // ----------member data ---------------------------
      
      std::vector<std::string> theCaloList;
};

