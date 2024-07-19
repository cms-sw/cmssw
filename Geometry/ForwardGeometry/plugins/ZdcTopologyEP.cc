// -*- C++ -*-
//
// Package:    ZdcTopologyEP
// Class:      ZdcTopologyEP
//
/**\class ZdcTopologyEP ZdcTopologyEP.h tmp/ZdcTopologyEP/interface/ZdcTopologyEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

#include "Geometry/ForwardGeometry/plugins/ZdcTopologyEP.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#define EDM_ML_DEBUG

ZdcTopologyEP::ZdcTopologyEP(const edm::ParameterSet& conf)
    : m_hdcToken{setWhatProduced(this, &ZdcTopologyEP::produce).consumes<HcalDDDRecConstants>(edm::ESInputTag{})},
      m_restrictions(conf.getUntrackedParameter<std::string>("Exclude")) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "ZdcTopologyEP::ZdcTopologyEP with Exclude: " << m_restrictions;
#endif
}

void ZdcTopologyEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("Exclude", "");
  descriptions.addWithDefaultLabel(desc);
}

// ------------ method called to produce the data  ------------
ZdcTopologyEP::ReturnType ZdcTopologyEP::produce(const HcalRecNumberingRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "ZdcTopologyEP::produce(const HcalRecNumberingRecord& iRecord)";
#endif
  const HcalDDDRecConstants& hdc = iRecord.get(m_hdcToken);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "mode = " << hdc.getTopoMode();
#endif
  ReturnType myTopo(new ZdcTopology(&hdc));
  return myTopo;
}
