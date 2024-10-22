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
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

//
// class decleration
//

class EcalTBGeometryBuilder : public edm::ESProducer {
public:
  EcalTBGeometryBuilder(const edm::ParameterSet&);
  ~EcalTBGeometryBuilder() override;

  typedef std::unique_ptr<CaloGeometry> ReturnType;

  ReturnType produce(const IdealGeometryRecord&);

private:
  // ----------member data ---------------------------
  edm::ESGetToken<CaloSubdetectorGeometry, IdealGeometryRecord> barrelToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, IdealGeometryRecord> hodoscopeToken_;
};
