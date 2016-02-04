// -*- C++ -*-
//
// Package:    EcalTBGeometryBuilder
// Class:      EcalTBGeometryBuilder
// 
/**\class EcalTBGeometryBuilder EcalTBGeometryBuilder.h tmp/EcalTBGeometryBuilder/interface/EcalTBGeometryBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
// $Id: EcalTBGeometryBuilder.h,v 1.1 2007/04/15 23:16:29 wmtan Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

//
// class decleration
//

class EcalTBGeometryBuilder : public edm::ESProducer {
   public:
  EcalTBGeometryBuilder(const edm::ParameterSet&);
  ~EcalTBGeometryBuilder();

  typedef std::auto_ptr<CaloGeometry> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);
private:
      // ----------member data ---------------------------
};

