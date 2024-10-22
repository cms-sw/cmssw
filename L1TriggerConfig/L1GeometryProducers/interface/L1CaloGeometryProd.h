#ifndef L1GeometryProducers_L1CaloGeometryProd_h
#define L1GeometryProducers_L1CaloGeometryProd_h
// -*- C++ -*-
//
// Package:     L1GeometryProducers
// Class  :     L1CaloGeometryProd
//
/**\class L1CaloGeometryProd L1CaloGeometryProd.h
 L1TriggerConfig/L1GeometryProducers/interface/L1CaloGeometryProd.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Tue Oct 24 00:01:12 EDT 2006
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"
#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"

// forward declarations

class L1CaloGeometryProd : public edm::ESProducer {
public:
  L1CaloGeometryProd(const edm::ParameterSet &);
  ~L1CaloGeometryProd() override;

  typedef std::unique_ptr<L1CaloGeometry> ReturnType;

  ReturnType produce(const L1CaloGeometryRecord &);

private:
  // ----------member data ---------------------------
  L1CaloGeometry m_geom;
};

#endif
