#ifndef CalibTracker_SiPhase2TrackerESProducers_SiPhase2OuterTrackerFakeLorentzAngleESSource_h
#define CalibTracker_SiPhase2TrackerESProducers_SiPhase2OuterTrackerFakeLorentzAngleESSource_h
// -*- C++ -*-
//
// Package:    SiPhase2OuterTrackerFakeLorentzAngleESSource
// Class:      SiPhase2OuterTrackerFakeLorentzAngleESSource
//
/**\class SiPhase2OuterTrackerFakeLorentzAngleESSource SiPhase2OuterTrackerFakeLorentzAngleESSource.h CalibTracker/SiPhase2TrackerESProducers/src/SiPhase2OuterTrackerFakeLorentzAngleESSource.cc
 Description: <one line class summary>
 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Marco Musich
//         Created:  July 31st, 2020
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/SiPhase2OuterTrackerLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerLorentzAngleRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//
// class decleration
//

class SiPhase2OuterTrackerFakeLorentzAngleESSource : public edm::ESProducer,
                                                     public edm::EventSetupRecordIntervalFinder {
public:
  SiPhase2OuterTrackerFakeLorentzAngleESSource(const edm::ParameterSet &);
  ~SiPhase2OuterTrackerFakeLorentzAngleESSource() override;

  void produce(){};
  static void fillDescriptions(edm::ConfigurationDescriptions &);

protected:
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &,
                      const edm::IOVSyncValue &,
                      edm::ValidityInterval &) override;

  virtual std::unique_ptr<SiPhase2OuterTrackerLorentzAngle> produceOTLA(const SiPhase2OuterTrackerLorentzAngleRcd &);
  virtual std::unique_ptr<SiPhase2OuterTrackerLorentzAngle> produceOTSimLA(
      const SiPhase2OuterTrackerLorentzAngleSimRcd &);

private:
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_tTopoToken;
  edm::ESGetToken<GeometricDet, IdealGeometryRecord> m_geomDetToken;
  const float LAvalue_;
  const std::string recordName_;
};
#endif
