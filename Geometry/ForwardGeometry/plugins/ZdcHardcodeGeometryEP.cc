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
//
#include "Geometry/Records/interface/ZDCGeometryRecord.h"
#include "Geometry/ForwardGeometry/plugins/ZdcHardcodeGeometryEP.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

ZdcHardcodeGeometryEP::ZdcHardcodeGeometryEP(const edm::ParameterSet& ps)
    : m_loader(nullptr),
      m_applyAlignment(ps.getParameter<bool>("applyAlignment")),
      m_zdcAddRPD(ps.getParameter<bool>("zdcAddRPD")) {
  //the following line is needed to tell the framework what
  // data is being produced
  //auto cc = setWhatProduced(this, &ZdcHardcodeGeometryEP::produce, ZdcGeometry::producerTag());
  auto cc = setWhatProduced(this, ZdcGeometry::producerTag());
  m_zdcTopoToken = cc.consumesFrom<ZdcTopology, HcalRecNumberingRecord>(edm::ESInputTag{});
  // disable
  //   setWhatProduced( this,
  //		    &ZdcHardcodeGeometryEP::produceIdeal,
  //		    edm::es::Label( "ZDC" ) );
}

ZdcHardcodeGeometryEP::~ZdcHardcodeGeometryEP() {}

//
// member functions
//

// ------------ method called to produce the data  ------------

ZdcHardcodeGeometryEP::ReturnType ZdcHardcodeGeometryEP::produce(const ZDCGeometryRecord& iRecord) {
  //   ZdcHardcodeGeometryLoader loader ( m_topology ) ;
  const ZdcTopology& m_topology = iRecord.get(m_zdcTopoToken);
  m_loader = std::make_unique<ZdcHardcodeGeometryLoader>(m_topology);
  m_loader->setAddRPD(m_zdcAddRPD);
  return ReturnType(m_loader->load());
}

void ZdcHardcodeGeometryEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("applyAlignment", false);
  desc.add<bool>("zdcAddRPD", false);
  descriptions.addWithDefaultLabel(desc);
}
