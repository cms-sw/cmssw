// -*- C++ -*-
//
// Package:    HcalTopologyIdealEP
// Class:      HcalTopologyIdealEP
//
/**\class HcalTopologyIdealEP HcalTopologyIdealEP.h tmp/HcalTopologyIdealEP/interface/HcalTopologyIdealEP.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
//
//

#include "Geometry/HcalEventSetup/interface/HcalTopologyIdealEP.h"
#include "Geometry/CaloTopology/interface/HcalTopologyRestrictionParser.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//#define EDM_ML_DEBUG

HcalTopologyIdealEP::HcalTopologyIdealEP(const edm::ParameterSet& conf)
    : m_hdcToken{setWhatProduced(this, &HcalTopologyIdealEP::produce).consumes<HcalDDDRecConstants>(edm::ESInputTag{})},
      m_restrictions(conf.getUntrackedParameter<std::string>("Exclude")),
      m_mergePosition(conf.getUntrackedParameter<bool>("MergePosition")) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalTopologyIdealEP::HcalTopologyIdealEP with Exclude: " << m_restrictions
                               << " MergePosition: " << m_mergePosition;
  edm::LogInfo("HCAL") << "HcalTopologyIdealEP::HcalTopologyIdealEP";
#endif
}

void HcalTopologyIdealEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("Exclude", "");
  desc.addUntracked<bool>("MergePosition", false);
  descriptions.add("hcalTopologyIdealBase", desc);
}

// ------------ method called to produce the data  ------------
HcalTopologyIdealEP::ReturnType HcalTopologyIdealEP::produce(const HcalRecNumberingRecord& iRecord) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "HcalTopologyIdealEP::produce(const IdealGeometryRecord& iRecord)";
  edm::LogInfo("HCAL") << "HcalTopologyIdealEP::produce(const HcalGeometryRecord& iRecord)";
#endif
  const HcalDDDRecConstants& hdc = iRecord.get(m_hdcToken);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "mode = " << hdc.getTopoMode() << ", maxDepthHB = " << hdc.getMaxDepth(0)
                               << ", maxDepthHE = " << hdc.getMaxDepth(1) << ", maxDepthHF = " << hdc.getMaxDepth(2);
  edm::LogInfo("HCAL") << "mode = " << hdc.getTopoMode() << ", maxDepthHB = " << hdc.getMaxDepth(0)
                       << ", maxDepthHE = " << hdc.getMaxDepth(1) << ", maxDepthHF = " << hdc.getMaxDepth(2);
#endif
  ReturnType myTopo(new HcalTopology(&hdc, m_mergePosition));

  HcalTopologyRestrictionParser parser(*myTopo);
  if (!m_restrictions.empty()) {
    std::string error = parser.parse(m_restrictions);
    if (!error.empty()) {
      throw cms::Exception("Parse Error", "Parse error on Exclude " + error);
    }
  }
  return myTopo;
}
